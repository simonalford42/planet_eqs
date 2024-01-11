
    def generate_f1_inputs_and_targets(self, batch, batch_idx):
        X_sample, y_sample = batch
        inputs, summary_stats = self.forward_to_summary_only(X_sample, noisy_val=False)
        return inputs, summary_stats

    def training_step(self, batch, batch_idx):
        fraction = self.global_step / self.hparams['steps']
        beta_in = min([1, fraction/0.3]) * self.beta_in
        beta_out = min([1, fraction/0.3]) * self.beta_out

        X_sample, y_sample = batch
        loss = self.lossfnc(X_sample, y_sample, noisy_val=True)
        #cur_frac = len(X_sample) / self.train_len

        # Want to be important with total number of samples
        input_kl = self.input_kl() * beta_in * len(X_sample)
        summary_kl = self.summary_kl() * beta_out

        prior = input_kl + summary_kl

        # total_loss = loss + prior
        total_loss = loss

        tensorboard_logs = {'train_loss_no_reg': loss/len(X_sample),
                            'train_loss_with_reg': total_loss/len(X_sample),
                            'input_kl': input_kl/len(X_sample),
                            'summary_kl': summary_kl/len(X_sample)}

        return {'loss': total_loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        X_sample, y_sample = batch
        loss = self.lossfnc(X_sample, y_sample, noisy_val=self.hparams['noisy_val'])/self.test_len

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):

        avg_loss = torch.stack([x['val_loss'] for x in outputs]).sum()

        tensorboard_logs = {'val_loss_no_reg': avg_loss}

        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        opt1 = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.hparams['momentum'], weight_decay=self.hparams['weight_decay'])

        assert self.hparams['scheduler_choice'] == 'swa'
        scheduler = CustomOneCycleLR(opt1, self.lr, int(0.9*self.steps), final_div_factor=1e4)
        interval = 'steps'
        name = 'swa_lr'

        sched1 = {
            'scheduler': scheduler,
            'name': name,
            'interval': interval
        }

        return [opt1], [sched1]

    def make_dataloaders(self, train=True, **extra_kwargs):
        kwargs = {
            **self.hparams,
            'model': self,
            **extra_kwargs,
            'train': train,
        }
        if 'ssX' in kwargs:
            dataloader, val_dataloader = get_data(**kwargs)
        else:
            dataloader, val_dataloader = get_data(ssX=self.ssX, **kwargs)

        labels = ['time', 'e+_near', 'e-_near', 'max_strength_mmr_near', 'e+_far', 'e-_far', 'max_strength_mmr_far', 'megno', 'a1', 'e1', 'i1', 'cos_Omega1', 'sin_Omega1', 'cos_pomega1', 'sin_pomega1', 'cos_theta1', 'sin_theta1', 'a2', 'e2', 'i2', 'cos_Omega2', 'sin_Omega2', 'cos_pomega2', 'sin_pomega2', 'cos_theta2', 'sin_theta2', 'a3', 'e3', 'i3', 'cos_Omega3', 'sin_Omega3', 'cos_pomega3', 'sin_pomega3', 'cos_theta3', 'sin_theta3', 'm1', 'm2', 'm3', 'nan_mmr_near', 'nan_mmr_far', 'nan_megno']
        for i in range(len(labels)):
            label = labels[i]
            if not ('cos' in label or
                'sin' in label or
                'nan_' in label or
                label == 'i1' or
                label == 'i2' or
                label == 'i3'):
                continue

            if not self.include_angles:
                print('Tossing', i, label)
                dataloader.dataset.tensors[0][..., i] = 0.0
                val_dataloader.dataset.tensors[0][..., i] = 0.0

        self._dataloader = dataloader
        self._val_dataloader = val_dataloader
        self.train_len = len(dataloader.dataset.tensors[0])
        self.test_len = len(val_dataloader.dataset.tensors[0])

    def train_dataloader(self):
        if self._dataloader is None:
            self.make_dataloaders()
        return self._dataloader

    def val_dataloader(self):
        if self._val_dataloader is None:
            self.make_dataloaders()
        return self._val_dataloader


class SWAGModel(VarModel):
    """Use .load_from_checkpoint(checkpoint_path) to initialize a SWAG model"""
    def init_params(self, swa_params):
        self.swa_params = swa_params
        self.swa_params['swa_lr'] = 0.001 if 'swa_lr' not in self.swa_params else self.swa_params['swa_lr']
        self.swa_params['swa_start'] = 1000 if 'swa_start' not in self.swa_params else self.swa_params['swa_start']
        self.swa_params['swa_recording_lr_factor'] = 0.5 if 'swa_recording_lr_factor' not in self.swa_params else self.swa_params['swa_recording_lr_factor']

        self.n_models = 0
        self.w_avg = None
        self.w2_avg = None
        self.pre_D = None
        self.K = 20 if 'K' not in self.swa_params else self.swa_params['K']
        self.c = 2 if 'c' not in self.swa_params else self.swa_params['c']
        self.swa_params['c'] = self.c
        self.swa_params['K'] = self.K

        return self

    def configure_optimizers(self):
        opt1 = torch.optim.SGD(self.parameters(), lr=self.swa_params['swa_lr'], momentum=self.hparams['momentum'], weight_decay=self.hparams['weight_decay'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt1, [self.swa_params['swa_start']], self.swa_params['swa_recording_lr_factor'])
        interval = 'steps'
        name = 'swa_record_lr'
        sched1 = {
            'scheduler': scheduler,
            'name': name,
            'interval': interval
        }

        return [opt1], [sched1]

    def training_step(self, batch, batch_idx):
        beta_in = self.beta_in
        beta_out = self.beta_out
        X_sample, y_sample = batch
        loss = self.lossfnc(X_sample, y_sample, noisy_val=True)
        input_kl = self.input_kl() * beta_in * len(X_sample)
        summary_kl = self.summary_kl() * beta_out
        prior = input_kl + summary_kl
        total_loss = loss + prior
        tensorboard_logs = {'train_loss_no_reg': loss/len(X_sample), 'train_loss_with_reg': total_loss/len(X_sample), 'input_kl': input_kl/len(X_sample), 'summary_kl': summary_kl/len(X_sample)}
        return {'loss': total_loss, 'log': tensorboard_logs}

    def flatten(self):
        """Convert state dict into a vector"""
        ps = self.state_dict()
        p_vec = None
        for key in ps.keys():
            p = ps[key]

            if p_vec is None:
                p_vec = p.reshape(-1)
            else:
                p_vec = torch.cat((p_vec, p.reshape(-1)))

        return p_vec

    def load(self, p_vec):
        """Load a vector into the state dict"""
        cur_state_dict = self.state_dict()
        new_state_dict = OrderedDict()
        i = 0
        for key in cur_state_dict.keys():
            old_p = cur_state_dict[key]
            size = old_p.numel()
            shape = old_p.shape
            new_p = p_vec[i:i+size]
            if len(shape) > 0:
                new_p = new_p.reshape(*shape)
            new_state_dict[key] = new_p
            i += size

        self.load_state_dict(new_state_dict)

    def aggregate_model(self):
        """Aggregate models for SWA/SWAG"""

        cur_w = self.flatten()
        cur_w2 = cur_w ** 2
        with torch.no_grad():
            if self.w_avg is None:
                self.w_avg = cur_w
                self.w2_avg = cur_w2
            else:
                self.w_avg = (self.w_avg * self.n_models + cur_w) / (self.n_models + 1)
                self.w2_avg = (self.w2_avg * self.n_models + cur_w2) / (self.n_models + 1)

            if self.pre_D is None:
                self.pre_D = cur_w.clone()[:, None]
            elif self.current_epoch % self.c == 0:
                #Record weights, measure discrepancy with average later
                self.pre_D = torch.cat((self.pre_D, cur_w[:, None]), dim=1)
                if self.pre_D.shape[1] > self.K:
                    self.pre_D = self.pre_D[:, 1:]


        self.n_models += 1

    def validation_step(self, batch, batch_idx):
        X_sample, y_sample = batch
        loss = self.lossfnc(X_sample, y_sample, noisy_val=self.hparams['noisy_val'])/self.test_len

        if self.w_avg is None:
            swa_loss = loss
        else:
            tmp = self.flatten()
            self.load(self.w_avg)
            swa_loss = self.lossfnc(X_sample, y_sample, noisy_val=self.hparams['noisy_val'])/self.test_len
            self.load(tmp)

        return {'val_loss': loss, 'swa_loss': swa_loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).sum()
        swa_avg_loss = torch.stack([x['swa_loss'] for x in outputs]).sum()
        tensorboard_logs = {'val_loss_no_reg': avg_loss, 'swa_loss_no_reg': swa_avg_loss}
        #TODO: Check
        #fraction = self.global_step / self.hparams['steps']
        #if fraction > 0.5:
        if self.global_step > self.hparams['swa_start']:
            self.aggregate_model()

        # Record validation loss, and aggregated model loss
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def sample_weights(self, scale=1):
        """Sample weights using SWAG:
        - w ~ N(avg_w, 1/2 * sigma + D . D^T/2(K-1))
            - This can be done with the following matrices:
                - z_1 ~ N(0, I_d); d the number of parameters
                - z_2 ~ N(0, I_K)
            - Then, compute:
            - w = avg_w + (1/sqrt(2)) * sigma^(1/2) . z_1 + D . z_2 / sqrt(2(K-1))
        """
        with torch.no_grad():
            avg_w = self.w_avg #[K]
            avg_w2 = self.w2_avg #[K]
            D = self.pre_D - avg_w[:, None]#[d, K]
            d = avg_w.shape[0]
            K = self.K
            z_1 = torch.randn((1, d), device=self.device)
            z_2 = torch.randn((K, 1), device=self.device)
            sigma = torch.abs(torch.diag(avg_w2 - avg_w**2))

            w = avg_w[None] + scale * (1.0/np.sqrt(2.0)) * z_1 @ sigma**0.5
            w += scale * (D @ z_2).T / np.sqrt(2*(K-1))
            w = w[0]

        self.load(w)

    def forward_swag(self, x, scale=0.5):
        """No augmentation happens here."""

        # Sample using SWAG using recorded model moments
        self.sample_weights(scale=scale)

        if self.fix_megno or self.fix_megno2:
            if self.fix_megno:
                megno_avg_std = self.summarize_megno(x)
            #(batch, 2)
            x = self.zero_megno(x)

        if not self.include_mmr:
            x = self.zero_mmr(x)

        if not self.include_nan:
            x = self.zero_nan(x)

        if not self.include_eplusminus:
            x = self.zero_eplusminus(x)

        summary_stats = self.compute_summary_stats(x)
        if self.fix_megno:
            summary_stats = torch.cat([summary_stats, megno_avg_std], dim=1)

        #summary is (batch, feature)
        self._summary_kl = (2/2) * (
                summary_stats**2
                + torch.exp(self.summary_noise_logvar)[None, :]
                - self.summary_noise_logvar[None, :]
                - 1
            )

        mu, std = self.predict_instability(summary_stats)
        #Each is (batch,)

        return torch.cat((mu, std), dim=1)

    def forward_swag_fast(self, x, scale=0.5):
        """No augmentation happens here."""

        # Sample using SWAG using recorded model moments
        self.sample_weights(scale=scale)

        if self.fix_megno or self.fix_megno2:
            if self.fix_megno:
                megno_avg_std = self.summarize_megno(x)
            #(batch, 2)
            x = self.zero_megno(x)

        if not self.include_mmr:
            x = self.zero_mmr(x)

        if not self.include_nan:
            x = self.zero_nan(x)

        if not self.include_eplusminus:
            x = self.zero_eplusminus(x)

        summary_stats = self.compute_summary_stats(x)
        if self.fix_megno:
            summary_stats = torch.cat([summary_stats, megno_avg_std], dim=1)

        #summary is (batch, feature)

        mu, std = self.predict_instability(summary_stats)
        #Each is (batch,)

        return torch.cat((mu, std), dim=1)


def save_swag(swag_model, path):
    save_items = {
        'hparams':swag_model.hparams,
        'swa_params': swag_model.swa_params,
        'w_avg': swag_model.w_avg.cpu(),
        'w2_avg': swag_model.w2_avg.cpu(),
        'pre_D': swag_model.pre_D.cpu()
    }

    torch.save(save_items, path)

def load_swag(path):
    save_items = torch.load(path)
    swag_model = (
        SWAGModel(save_items['hparams'])
        .init_params(save_items['swa_params'])
    )
    swag_model.w_avg = save_items['w_avg']
    swag_model.w2_avg = save_items['w2_avg']
    swag_model.pre_D = save_items['pre_D']
    if 'v50' in path:
        # Assume fixed scale:
        ssX = StandardScaler()
        ssX.scale_ = np.array([2.88976974e+03, 6.10019661e-02, 4.03849732e-02, 4.81638693e+01,
                   6.72583662e-02, 4.17939679e-02, 8.15995339e+00, 2.26871589e+01,
                   4.73612029e-03, 7.09223721e-02, 3.06455099e-02, 7.10726478e-01,
                   7.03392022e-01, 7.07873597e-01, 7.06030923e-01, 7.04728204e-01,
                   7.09420909e-01, 1.90740659e-01, 4.75502285e-02, 2.77188320e-02,
                   7.08891412e-01, 7.05214134e-01, 7.09786887e-01, 7.04371833e-01,
                   7.04371110e-01, 7.09828420e-01, 3.33589977e-01, 5.20857790e-02,
                   2.84763136e-02, 7.02210626e-01, 7.11815232e-01, 7.10512240e-01,
                   7.03646004e-01, 7.08017286e-01, 7.06162814e-01, 2.12569430e-05,
                   2.35019125e-05, 2.04211110e-05, 7.51048890e-02, 3.94254400e-01,
                   7.11351099e-02])
        ssX.mean_ = np.array([ 4.95458585e+03,  5.67411891e-02,  3.83176945e-02,  2.97223474e+00,
                   6.29733979e-02,  3.50074471e-02,  6.72845676e-01,  9.92794768e+00,
                   9.99628430e-01,  5.39591547e-02,  2.92795061e-02,  2.12480714e-03,
                  -1.01500319e-02,  1.82667162e-02,  1.00813201e-02,  5.74404197e-03,
                   6.86570242e-03,  1.25316320e+00,  4.76946516e-02,  2.71326280e-02,
                   7.02054326e-03,  9.83378673e-03, -5.70616748e-03,  5.50782881e-03,
                  -8.44213953e-04,  2.05958338e-03,  1.57866569e+00,  4.31476211e-02,
                   2.73316392e-02,  1.05505555e-02,  1.03922250e-02,  7.36865006e-03,
                  -6.00523246e-04,  6.53016990e-03, -1.72038113e-03,  1.24807860e-05,
                   1.60314173e-05,  1.21732696e-05,  5.67292645e-03,  1.92488263e-01,
                   5.08607199e-03])
        ssX.var_ = ssX.scale_**2
        swag_model.ssX = ssX
    else:
        ssX_file = path[:-4] + '_ssX.pkl'
        try:
            ssX = pkl.load(open(ssX_file, 'rb'))
            swag_model.ssX = ssX
        except FileNotFoundError:
            print(f"ssX file not found! {ssX_file}")
            ...

    return swag_model
