# coding=utf-8
import datetime
import torch
torch.manual_seed(97)
from dataloader import DataGenerator
from model import FlightBERT_PP
from utils import (load_config_from_json, print_attrs, file_print, convert2binfromlist,
                   calculate_max_distance, calculate_trajectory_frechet_distance, 
                   calculate_trajectory_dtw_distance, calculate_trajectory_dtw_distance_whole)
import os
import tqdm
from torch.utils.data import DataLoader
import numpy as np
import time
import argparse
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

data_worker = 0
iscuda = torch.cuda.is_available()
gpuid = 'cpu'

if iscuda:
    # torch.backends.cudnn.benchmark = True
    gpuid = torch.cuda.current_device()
    print('gpuid', gpuid)

available_devices = range(torch.cuda.device_count())
device = torch.device("cuda:{}".format(gpuid) if iscuda else 'cpu')
print('iscuda', iscuda)

def save_torch_model(model, opt, save_path):
    checkpt = {'model_state_dict': model.state_dict() if len(available_devices) <= 1 else model.module.state_dict(),
               'optimizer_state_dict': opt.state_dict()}
    torch.save(checkpt, save_path)


def load_torch_model(model, optimizer, model_path):
    map_location = None if iscuda else 'cpu'
    ckpt = torch.load(model_path, map_location=map_location)
    model.load_state_dict(ckpt['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

def worker_init_fn(worker_id):
    np.random.seed(7 + worker_id)


class TrainClient():
    def __init__(self, json_path='config.json'):
        print("Load configs")
        self.configs = load_config_from_json(json_path)
        ATTRS = print_attrs(self.configs)
        self.log_path = self.configs.save_dir
        self.log_path = os.path.join(self.log_path, str(datetime.date.today()))
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        with open(os.path.join(self.log_path, 'readme'), 'w+') as fw:
            fw.write(ATTRS)

        self.train_datas = self.configs.train_data
        self.dev_datas = self.configs.dev_data
        self.test_datas = self.configs.test_data
        self.log_file = "train.log" if self.configs.is_training else "test.log"
        self.logger(str(self.configs), self.log_file)
        self.train_costs, self.val_costs = [], []
        self.vocab_size = 0
        self.keep_model_list = []
        # Add unique run id for each tensorboard run
        run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tensorboard_dir = getattr(self.configs, 'tensorboard_dir', 'tensorboard')
        tensorboard_filename = getattr(self.configs, 'tensorboard_filename', run_id)
        self.writer = SummaryWriter(log_dir=os.path.join(self.log_path, tensorboard_dir, tensorboard_filename))

        # Log learning configuration to tensorboard
        config_text = '\n'.join([f'{k}: {v}' for k, v in self.configs.__dict__.items() if not k.startswith('__')])
        self.writer.add_text('Learning Configuration', config_text, 0)

    def load_data(self):
        print("Load data...")
        self.eval_gen = None
        self.train_gen = None
        if self.configs.is_training:
            self.train_gen = DataGenerator(self.configs)
            self.train_gen.load_data_from_dir(self.train_datas)

            self.eval_gen = DataGenerator(self.configs)
            self.eval_gen.load_data_from_dir(self.dev_datas)
        else:
            self.eval_gen = DataGenerator(self.configs)
            self.eval_gen.load_data_from_dir(self.test_datas)

    def init_model(self):
        print("Build model...")
        self.model = FlightBERT_PP(config=self.configs)
        self.model.to(device)

        total_pa = sum(p.numel() for p in self.model.parameters())
        print("Total params: %.2f M" % (total_pa / 1e6))

        # loss function ...
        self.BCEloss = torch.nn.BCELoss()

    def run(self):
        self.load_data()
        self.init_model()

        if self.configs.is_training:
            self.run_train()
        else:
            model_path = self.configs.model_path
            if model_path != '' or (not self.configs.is_training):
                self.logger("Test: Reload params from " + model_path, self.log_file)
                load_torch_model(self.model, model_path)

            self.run_test(999, 0, self.configs.batch_size, full_batch=True)

    def run_train(self):
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=self.configs.learning_rate)
        self.opt_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimiser, step_size=20, gamma=0.7)
        model_path = self.configs.model_path
        start_epoch = 0

        if model_path != '':
            self.logger("Reload params from " + model_path, self.log_file)
            load_torch_model(self.model, self.optimiser, model_path)
            # Try to extract epoch from filename if possible
            import re
            match = re.search(r'epoch_(\\d+)_', os.path.basename(model_path))
            if match:
                start_epoch = int(match.group(1)) + 1

        for p in self.model.parameters():
            p.requires_grad = True

        print('model ....')
        print(self.model)

        print('train process...')
        for epoch in range(start_epoch, self.configs.epochs):
            self.run_train_epoch(epoch, self.configs.batch_size)
            self.opt_lr_scheduler.step()
            print('training lr at epoch {} is {}'.format(epoch, self.opt_lr_scheduler.get_last_lr()[0]))

    def logger(self, info, log_file, debug=True):
        file_print(info, logfilename=log_file, savepath=self.log_path, debug=debug)

    def run_train_epoch(self, epoch, batch_size=16):
        self.model.train()
        self.logger("Run the training epoch {}\n".format(epoch), log_file=self.log_file)

        num_batches = self.train_gen.data_num // batch_size

        data_loader = DataLoader(self.train_gen, batch_size=batch_size, shuffle=True,
                                 num_workers=data_worker, collate_fn=self.train_gen.prepare_minibatch,
                                 pin_memory=iscuda, worker_init_fn=worker_init_fn)

        tq = tqdm.tqdm(iter(data_loader), desc='Training epoch {}'.format(epoch), total=num_batches, dynamic_ncols=True)

        start_time = time.time()
        epoch_loss_sum = 0.0
        epoch_loss_count = 0
        for i, batch in enumerate(tq):
            batch_tmp = {}
            for k, v in batch.items():
                v = torch.FloatTensor(np.array(v))
                batch_tmp[k] = v.to(device)
            batch = batch_tmp

            lon_inp, lat_inp, alt_inp, spdx_inp, spdy_inp, spdz_inp = batch['lon'], batch['lat'], batch['alt'], \
                                                                      batch['spdx'], batch['spdy'], \
                                                                      batch['spdz']

            t_lon, t_lat, t_alt, t_spdx, t_spdy, t_spdz = batch['t_lon'], batch['t_lat'], batch['t_alt'], \
                                                          batch['t_spdx'], batch['t_spdy'], batch['t_spdz']

            raw_lon, raw_lat, raw_alt, raw_spdx, raw_spdy, raw_pdz = batch['raw_lon'], batch['raw_lat'], \
                                                                     batch['raw_alt'], batch['raw_spdx'], batch[
                                                                         'raw_spdy'], batch['raw_spdz']

            dec_lon, dec_lat, dec_alt, dec_spdx, dec_spdy, dec_spdz = batch['dec_lon'], batch['dec_lat'], \
                                                                      batch['dec_alt'], batch['dec_spdx'], \
                                                                      batch['dec_spdy'], batch['dec_spdz']

            raw_data = raw_lon, raw_lat, raw_alt, raw_spdx, raw_spdy, raw_pdz
            target_items = t_lon, t_lat, t_alt, t_spdx, t_spdy, t_spdz
            dec_inputs = torch.cat((dec_lon, dec_lat, dec_alt, dec_spdx, dec_spdy, dec_spdz), dim=-1)

            # train  ...
            lon_logits, lat_logits, alt_logits, spdx_logits, spdy_logits, spdz_logits = self.model(lon_inp, lat_inp,
                                                                                                   alt_inp, spdx_inp,
                                                                                                   spdy_inp, spdz_inp,
                                                                                                   dec_inputs)

            # calculate loss and backword ...
            model_outputs = lon_logits, lat_logits, alt_logits, spdx_logits, spdy_logits, spdz_logits
            bce_loss, details, cvt_values = self.calculate_loss(model_outputs, target_items, raw_data,
                                                                      mode='train')

            self.optimiser.zero_grad()
            bce_loss.backward()
            self.optimiser.step()

            epoch_loss_sum += bce_loss.item()
            epoch_loss_count += 1

            strprint = "loss at epoch {}, iter {}/{},bce:{:.4f}, time cost={}".format(
                epoch, i, num_batches, bce_loss, round(time.time() - start_time, 2))

            self.logger(strprint, log_file=self.log_file, debug=False)
            start_time = time.time()  # <-- this is per batch, keep for batch timing

            tq.set_postfix_str(strprint)

        # Log mean loss for the epoch
        mean_loss = epoch_loss_sum / epoch_loss_count if epoch_loss_count > 0 else 0.0
        self.writer.add_scalar('Loss/train', mean_loss, epoch)

        # Log epoch duration
        epoch_time = time.time() - tq.start_t
        self.writer.add_scalar('Time/epoch_train', epoch_time, epoch)

        self.run_test(epoch, batch_size=self.configs.batch_size, full_batch=True)

    def run_test(self, epoch, batch_size=16, full_batch=False):
        print("test process...")
        self.model.eval()
        batch_num = self.eval_gen.data_num // batch_size if full_batch else 1

        self.logger('Run the model evaluation at Epoch:{}, batch_num:{}'.format(epoch, batch_num),
                    log_file=self.log_file)

        data_loader = DataLoader(self.eval_gen, batch_size=batch_size,
                                 num_workers=data_worker, collate_fn=self.eval_gen.prepare_minibatch,
                                 pin_memory=iscuda, worker_init_fn=worker_init_fn)

        losses = []
        avg_acc = []

        all_preds_cvt_values = {'lon': [], 'lat': [], 'alt': [], 'spdx': [], 'spdy': [], 'spdz': []}
        all_targs_cvt_values = {'lon': [], 'lat': [], 'alt': [], 'spdx': [], 'spdy': [], 'spdz': []}

        tq = tqdm.tqdm(enumerate(data_loader), desc='Testing epoch {}'.format(epoch), total=batch_num, dynamic_ncols=True)
        for i, batch in tq:
            start_time = time.time()
            
            batch_tmp = {}
            for k, v in batch.items():
                v = torch.FloatTensor(np.array(v))
                batch_tmp[k] = v.to(device)
            batch = batch_tmp
            input_items = batch['lon'], batch['lat'], batch['alt'], batch['spdx'], batch['spdy'], \
                          batch['spdz']

            with torch.no_grad():
                t_lon, t_lat, t_alt, t_spdx, t_spdy, t_spdz = batch['t_lon'], batch['t_lat'], batch['t_alt'], \
                                                              batch['t_spdx'], batch['t_spdy'], batch['t_spdz']

                raw_lon, raw_lat, raw_alt, raw_spdx, raw_spdy, raw_pdz = batch['raw_lon'], batch['raw_lat'], \
                                                                         batch['raw_alt'], batch['raw_spdx'], batch[
                                                                             'raw_spdy'], batch['raw_spdz']

                dec_lon, dec_lat, dec_alt, dec_spdx, dec_spdy, dec_spdz = batch['dec_lon'], batch['dec_lat'], \
                                                                          batch['dec_alt'], batch['dec_spdx'], \
                                                                          batch['dec_spdy'], batch['dec_spdz']

                raw_data = raw_lon, raw_lat, raw_alt, raw_spdx, raw_spdy, raw_pdz
                lon_inp, lat_inp, alt_inp, spdx_inp, spdy_inp, spdz_inp = input_items
                dec_inputs = torch.cat((dec_lon, dec_lat, dec_alt, dec_spdx, dec_spdy, dec_spdz), dim=-1)
                target_items = t_lon, t_lat, t_alt, t_spdx, t_spdy, t_spdz

                lon_logits, lat_logits, alt_logits, spdx_logits, spdy_logits, spdz_logits = self.model(lon_inp,
                                                                                                       lat_inp,
                                                                                                       alt_inp,
                                                                                                       spdx_inp,
                                                                                                       spdy_inp,
                                                                                                       spdz_inp,
                                                                                                       dec_inputs)

                # calculate loss ...
                model_outputs = lon_logits, lat_logits, alt_logits, spdx_logits, spdy_logits, spdz_logits

                loss, details, cvt_values = self.calculate_loss(model_outputs,target_items, raw_data, mode='test')

                for k, v in cvt_values.items():
                    all_preds_cvt_values[k].extend(v['preds_v'])
                    all_targs_cvt_values[k].extend(v['tarts_v'])

                losses.append(loss.item())

            acc = np.mean([i for i in details.values()])
            avg_acc.append(acc)

            if not full_batch:
                break

        for k, v in all_targs_cvt_values.items():
            all_targs_cvt_values[k] = np.array(v)

        for k, v in all_preds_cvt_values.items():
            all_preds_cvt_values[k] = np.array(v)

        # Log mean validation loss for the epoch
        mean_val_loss = np.mean(losses) if len(losses) > 0 else 0.0
        self.writer.add_scalar('Loss/val', mean_val_loss, epoch)

        evalute_horizons = [1, 3, 9] # range(1, self.configs.horizon + 1)
        for horizon in evalute_horizons:
            self.logger('Evaluation summary of Epoch:{}, Horizon:{}'.format(epoch, horizon), log_file=self.log_file)

            lon_rmse = np.sqrt(metrics.mean_squared_error(all_targs_cvt_values['lon'][:, :horizon],
                                                          all_preds_cvt_values['lon'][:, :horizon]))
            lat_rmse = np.sqrt(metrics.mean_squared_error(all_targs_cvt_values['lat'][:, :horizon],
                                                          all_preds_cvt_values['lat'][:, :horizon]))
            alt_rmse = np.sqrt(metrics.mean_squared_error(all_targs_cvt_values['alt'][:, :horizon],
                                                          all_preds_cvt_values['alt'][:, :horizon]))
            spdx_rmse = np.sqrt(metrics.mean_squared_error(all_targs_cvt_values['spdx'][:, :horizon],
                                                           all_preds_cvt_values['spdx'][:, :horizon]))
            spdy_rmse = np.sqrt(metrics.mean_squared_error(all_targs_cvt_values['spdy'][:, :horizon],
                                                           all_preds_cvt_values['spdy'][:, :horizon]))
            spdz_rmse = np.sqrt(metrics.mean_squared_error(all_targs_cvt_values['spdz'][:, :horizon],
                                                           all_preds_cvt_values['spdz'][:, :horizon]))

            print_all_rmse_details = 'lon:{},lat:{},alt:{},spdx:{},spdy:{},spdz:{}'.format(lon_rmse,
                                                                                           lat_rmse,
                                                                                           alt_rmse,
                                                                                           spdx_rmse,
                                                                                           spdy_rmse,
                                                                                           spdz_rmse)

            lon_mse = metrics.mean_absolute_error(all_targs_cvt_values['lon'][:, :horizon],
                                                  all_preds_cvt_values['lon'][:, :horizon])
            lat_mse = metrics.mean_absolute_error(all_targs_cvt_values['lat'][:, :horizon],
                                                  all_preds_cvt_values['lat'][:, :horizon])
            alt_mse = metrics.mean_absolute_error(all_targs_cvt_values['alt'][:, :horizon],
                                                  all_preds_cvt_values['alt'][:, :horizon])
            spdx_mse = metrics.mean_absolute_error(all_targs_cvt_values['spdx'][:, :horizon],
                                                   all_preds_cvt_values['spdx'][:, :horizon])
            spdy_mse = metrics.mean_absolute_error(all_targs_cvt_values['spdy'][:, :horizon],
                                                   all_preds_cvt_values['spdy'][:, :horizon])
            spdz_mse = metrics.mean_absolute_error(all_targs_cvt_values['spdz'][:, :horizon],
                                                   all_preds_cvt_values['spdz'][:, :horizon])

            print_mae_details = 'lon:{},lat:{},alt:{},spdx:{},spdy:{},spdz:{}'.format(lon_mse,
                                                                                      lat_mse,
                                                                                      alt_mse,
                                                                                      spdx_mse,
                                                                                      spdy_mse,
                                                                                      spdz_mse)

            lon_mape = metrics.mean_absolute_percentage_error(all_targs_cvt_values['lon'][:, :horizon],
                                                              all_preds_cvt_values['lon'][:, :horizon])
            lat_mape = metrics.mean_absolute_percentage_error(all_targs_cvt_values['lat'][:, :horizon],
                                                              all_preds_cvt_values['lat'][:, :horizon])

            alt_targ_array = np.array(all_targs_cvt_values['alt'][:, :horizon])
            logit = alt_targ_array.flatten() != 0
            alt_pred_array = np.array(all_preds_cvt_values['alt'][:, :horizon])
            alt_mape = metrics.mean_absolute_percentage_error(alt_targ_array.flatten()[logit],
                                                              alt_pred_array.flatten()[logit])

            print_mape_details = 'lon:{},lat:{},alt:{}'.format(lon_mape * 100, lat_mape * 100, alt_mape * 100)

            MDE = self.Cal_MDE(all_targs_cvt_values, all_preds_cvt_values, horizon)

            # Calculate new metrics
            # Ensure all values are numpy arrays and at least 2D before slicing
            def safe_slice(arr, horizon):
                arr = np.array(arr)
                if arr.ndim == 1:
                    return arr[:horizon]
                elif arr.ndim >= 2:
                    return arr[:, :horizon]
                return arr
            preds_h = {k: safe_slice(v, horizon) for k, v in all_preds_cvt_values.items()}
            targs_h = {k: safe_slice(v, horizon) for k, v in all_targs_cvt_values.items()}

            max_distance = calculate_max_distance(preds_h, targs_h)
            frechet_dist = calculate_trajectory_frechet_distance(preds_h, targs_h, horizon)
            dtw_distances = calculate_trajectory_dtw_distance(preds_h, targs_h, horizon)
            # Calculate DTW for the whole trajectory (up to horizon)
            dtw_whole = calculate_trajectory_dtw_distance_whole(preds_h, targs_h)
            
            dtw_summary = 'lon:{:.4f},lat:{:.4f},alt:{:.4f}'.format(
                dtw_distances['lon'], dtw_distances['lat'], dtw_distances['alt']
            )

            self.logger('Evaluation summary of Epoch:{}, batch:{}, MAE:{}'.format(
                epoch, batch_num, print_mae_details), log_file=self.log_file)

            self.logger('Evaluation summary of Epoch:{}, batch:{}, MAPE:{}'.format(
                epoch, batch_num, print_mape_details), log_file=self.log_file)

            self.logger('Evaluation summary of Epoch:{}, batch:{}, RMSE:{}'.format(
                epoch, batch_num, print_all_rmse_details), log_file=self.log_file)

            self.logger('Evaluation summary of Epoch:{}, batch:{}, MDE:{}'.format(
                epoch, batch_num, MDE), log_file=self.log_file)

            self.logger('Evaluation summary of Epoch:{}, batch:{}, MAX_DISTANCE:{:.4f}'.format(
                epoch, batch_num, max_distance), log_file=self.log_file)

            self.logger('Evaluation summary of Epoch:{}, batch:{}, FRECHET_DISTANCE:{:.4f}'.format(
                epoch, batch_num, frechet_dist), log_file=self.log_file)
            self.logger('Evaluation summary of Epoch:{}, batch:{}, DTW_DISTANCES:{}'.format(
                epoch, batch_num, dtw_summary), log_file=self.log_file)
            self.logger('Evaluation summary of Epoch:{}, batch:{}, DTW_WHOLE:{:.4f}'.format(
                epoch, batch_num, dtw_whole), log_file=self.log_file)

            self.writer.add_scalar(f'RMSE/lon_h{horizon}', lon_rmse, epoch)
            self.writer.add_scalar(f'RMSE/lat_h{horizon}', lat_rmse, epoch)
            self.writer.add_scalar(f'RMSE/alt_h{horizon}', alt_rmse, epoch)
            self.writer.add_scalar(f'RMSE/spdx_h{horizon}', spdx_rmse, epoch)
            self.writer.add_scalar(f'RMSE/spdy_h{horizon}', spdy_rmse, epoch)
            self.writer.add_scalar(f'RMSE/spdz_h{horizon}', spdz_rmse, epoch)
            self.writer.add_scalar(f'MAE/lon_h{horizon}', lon_mse, epoch)
            self.writer.add_scalar(f'MAE/lat_h{horizon}', lat_mse, epoch)
            self.writer.add_scalar(f'MAE/alt_h{horizon}', alt_mse, epoch)
            self.writer.add_scalar(f'MAE/spdx_h{horizon}', spdx_mse, epoch)
            self.writer.add_scalar(f'MAE/spdy_h{horizon}', spdy_mse, epoch)
            self.writer.add_scalar(f'MAE/spdz_h{horizon}', spdz_mse, epoch)
            self.writer.add_scalar(f'MAPE/lon_h{horizon}', lon_mape * 100, epoch)
            self.writer.add_scalar(f'MAPE/lat_h{horizon}', lat_mape * 100, epoch)
            self.writer.add_scalar(f'MAPE/alt_h{horizon}', alt_mape * 100, epoch)
            self.writer.add_scalar(f'MDE/h{horizon}', MDE, epoch)
            self.writer.add_scalar(f'MAX_DISTANCE/h{horizon}', max_distance, epoch)
            self.writer.add_scalar(f'FRECHET_DISTANCE/h{horizon}', frechet_dist, epoch)
            self.writer.add_scalar(f'DTW/lon_h{horizon}', dtw_distances['lon'], epoch)
            self.writer.add_scalar(f'DTW/lat_h{horizon}', dtw_distances['lat'], epoch)
            self.writer.add_scalar(f'DTW/alt_h{horizon}', dtw_distances['alt'], epoch)
            self.writer.add_scalar(f'DTW/whole_h{horizon}', dtw_whole, epoch)

        if self.configs.is_training:
            save_torch_model(self.model, self.optimiser,
                             self.log_path + "/epoch_{}_{}.pt".format(epoch, np.mean(avg_acc)))
            model_path = self.log_path + "/epoch_{}_{}.pt".format(epoch, np.mean(avg_acc))

            self.keep_model_list.append(model_path)

            if len(self.keep_model_list) > 2:
                rm = self.keep_model_list.pop(0)
                os.remove(rm)

    def gc2ecef(self, lon, lat, alt):
        a = 6378.137  # km
        b = 6356.752
        lat = np.radians(lat)
        lon = np.radians(lon)
        e_square = 1 - (b ** 2) / (a ** 2)
        N = a / np.sqrt(1 - e_square * (np.sin(lat) ** 2))
        X = (N + alt) * np.cos(lat) * np.cos(lon)
        Y = (N + alt) * np.cos(lat) * np.sin(lon)
        Z = ((b ** 2) / (a ** 2) * N + alt) * np.sin(lat)
        return X, Y, Z

    def Cal_MDE(self, pred, tart, horizon):
        mde_values = []
        for i in range(len(pred['lon'])):
            # Convert to ECEF coordinates (in km)
            X, Y, Z = self.gc2ecef(pred['lon'][i][:horizon], pred['lat'][i][:horizon],
                                   pred['alt'][i][:horizon] / 100)
            X_t, Y_t, Z_t = self.gc2ecef(tart['lon'][i][:horizon], tart['lat'][i][:horizon],
                                         tart['alt'][i][:horizon] / 100)
            mde = np.mean(np.sqrt((X - X_t) ** 2 + (Y - Y_t) ** 2 + (Z - Z_t) ** 2))
            mde_values.append(mde)
        MDE = np.mean(mde_values)
        return MDE

    def calculate_loss(self, model_outputs, target_items, raw_data, mode='train'):
        lon_logits, lat_logits, alt_logits, spdx_logits, spdy_logits, spdz_logits = model_outputs
        lon_targ, lat_targ, alt_targ, spdx_targ, spdy_targ, spdz_targ = target_items

        # Calculate the loss
        lon_loss = self.BCEloss(lon_logits.contiguous().view(-1, self.configs.delta_lon_size),
                                lon_targ.contiguous().view(-1, self.configs.delta_lon_size))

        lat_loss = self.BCEloss(lat_logits.contiguous().view(-1, self.configs.delta_lat_size),
                                lat_targ.contiguous().view(-1, self.configs.delta_lat_size))

        alt_loss = self.BCEloss(alt_logits.contiguous().view(-1, self.configs.delta_alt_size),
                                alt_targ.contiguous().view(-1, self.configs.delta_alt_size))

        spdx_loss = self.BCEloss(spdx_logits.contiguous().view(-1, self.configs.delta_spdx_size),
                                 spdx_targ.contiguous().view(-1, self.configs.delta_spdx_size))

        spdy_loss = self.BCEloss(spdy_logits.contiguous().view(-1, self.configs.delta_spdy_size),
                                 spdy_targ.contiguous().view(-1, self.configs.delta_spdy_size))

        spdz_loss = self.BCEloss(spdz_logits.contiguous().view(-1, self.configs.delta_spdz_size),
                                 spdz_targ.contiguous().view(-1, self.configs.delta_spdz_size))

        loss_tuple = (lat_loss, lon_loss, alt_loss, spdx_loss, spdy_loss, spdz_loss)

        loss = sum(loss_tuple)
        loss = loss.mean()

        if mode == 'train':
            return loss, None, None

        preds = [lon_logits, lat_logits, alt_logits, spdx_logits, spdy_logits, spdz_logits]
        trgts = [lon_targ, lat_targ, alt_targ, spdx_targ, spdy_targ, spdz_targ]
        
        # Measure execution time for evaluation
        accs, cvt_values = self.evalute_all(preds, trgts, raw_data)

        return loss, accs, cvt_values

    def evalute_all(self, preds, trgts, raw_data):
        desc = ('lon', 'lat', 'alt', 'spdx', 'spdy', 'spdz')
        accs = {}
        cvt_values = {'lon': {}, 'lat': {}, 'alt': {}, 'spdx': {}, 'spdy': {}, 'spdz': {}}

        for name, pred, trgt, raw in zip(desc, preds, trgts, raw_data):
            acc, preds_values, target_values = self.evalute(pred, trgt, name, raw)
            accs[name] = acc
            cvt_values[name]['preds_v'] = preds_values
            cvt_values[name]['tarts_v'] = target_values

        return accs, cvt_values

    def evalute(self, pred, trgt, name, raw):
        pred = (pred >= 0.5).int()
        trgt = trgt.int()

        batch_size, seq, dim = pred.size()
        acc = 0.0
        batch_cvt_preds = []
        batch_cvt_trgts = []

        # Move everything to cpu and numpy once
        pred_np = pred.cpu().numpy()
        trgt_np = trgt.cpu().numpy()
        raw_np = raw.cpu().numpy() if isinstance(raw, torch.Tensor) else np.array(raw)

        for i in range(batch_size):
            cvt_preds = []
            cvt_trgts = []
            last_pred = 0.0
            for j in range(seq):
                pred_list = pred_np[i, j].tolist()
                trgt_list = trgt_np[i, j].tolist()
                if pred_list == trgt_list:
                    acc += 1

                p_sign, t_sign = pred_list[0], trgt_list[0]
                cvt_pred = convert2binfromlist(pred_list[1:])
                # cvt_trgt = convert2binfromlist(trgt_list[1:])

                if p_sign == 1:
                    cvt_pred = -cvt_pred

                # if t_sign == 1:
                #     cvt_trgt = -cvt_trgt

                if j == 0:
                    real_pred = raw_np[i, self.configs.inp_seq_len + j - 1] + cvt_pred
                    last_pred = real_pred
                else:
                    real_pred = last_pred + cvt_pred
                    last_pred = real_pred

                real_targt = raw_np[i, self.configs.inp_seq_len + j]

                if name in ('lon', 'lat'):
                    real_pred = real_pred / 1000.0
                    # cvt_trgt = cvt_trgt / 1000.0
                    real_targt = real_targt / 1000.0

                cvt_preds.append(real_pred)
                cvt_trgts.append(real_targt)

            batch_cvt_preds.append(cvt_preds)
            batch_cvt_trgts.append(cvt_trgts)

        return acc / (batch_size * seq), np.array(batch_cvt_preds), np.array(batch_cvt_trgts)

    def close(self):
        self.writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="config path for the training", default='config.json',
                        required=False)
    args = parser.parse_args()
    tc = TrainClient(json_path=args.config)
    tc.run()
    tc.close()
