import os
import json
import random
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy import stats
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_angular_velocity(alpha_sequence, dt=0.001):

    if len(alpha_sequence) < 2:
        return np.zeros_like(alpha_sequence)

    omega = np.zeros_like(alpha_sequence)

    if len(alpha_sequence) > 2:
        omega[1:-1] = (alpha_sequence[2:] - alpha_sequence[:-2]) / (2 * dt)

    omega[0] = (alpha_sequence[1] - alpha_sequence[0]) / dt
    omega[-1] = (alpha_sequence[-1] - alpha_sequence[-2]) / dt

    return omega


def multi_step_sequence_predict_autoregressive(model, alpha_true, cond, norm, seq_len=16,
                                               output_len=8, device='cuda'):

    model.eval()

    a_mean, a_std = norm['a_mean'], norm['a_std']


    alpha_norm = (alpha_true - a_mean) / a_std

    total_steps = len(alpha_true)
    predictions = np.zeros(total_steps)


    predictions[:seq_len] = alpha_true[:seq_len]

    with torch.no_grad():

        current_idx = seq_len


        rolling_window = alpha_norm[:seq_len].copy()

        while current_idx < total_steps:


            current_seq = rolling_window[-seq_len:].copy()


            if len(current_seq) < seq_len:
                padding = np.full(seq_len - len(current_seq), current_seq[0])
                current_seq = np.concatenate([padding, current_seq])
            elif len(current_seq) > seq_len:
                current_seq = current_seq[-seq_len:]


            cond_seq = np.tile(cond, (seq_len, 1))
            input_seq = np.concatenate([
                current_seq.reshape(seq_len, 1),
                cond_seq
            ], axis=1)


            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)


            output = model(input_tensor)
            pred_norm = output[0].cpu().numpy()


            pred_denorm = pred_norm * a_std + a_mean


            steps_to_fill = min(output_len, total_steps - current_idx)
            predictions[current_idx:current_idx + steps_to_fill] = pred_denorm[:steps_to_fill]


            new_predictions_norm = (pred_denorm[:steps_to_fill] - a_mean) / a_std
            rolling_window = np.concatenate([rolling_window, new_predictions_norm])


            current_idx += output_len

    return predictions


class Seq2SeqAlphaDataset(Dataset):
    def __init__(self, data_dir, norm, folders, input_len=16, output_len=8, stride=5,
                 calc_norm=False, external_norm=None, teacher_forcing_ratio=0.5):

        self.input_len = input_len
        self.output_len = output_len
        self.stride = stride
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.samples = []

        if external_norm is not None:
            effective_norm = external_norm
        else:
            effective_norm = norm

        a_mean, a_std = effective_norm['a_mean'], effective_norm['a_std']
        u_mean, u_std = effective_norm['u_mean'], effective_norm['u_std']
        cm_mean, cm_std = effective_norm['cm_mean'], effective_norm['cm_std']

        if calc_norm:
            Ka_list, Ca_list, Ia_list = [], [], []
            for folder in folders:
                fp = os.path.join(data_dir, folder)
                if not os.path.isdir(fp):
                    continue
                try:
                    p = json.load(open(os.path.join(fp, 'params.json')))
                    Ka_list.append(p['Ka'])
                    Ca_list.append(p['Ca'])
                    Ia_list.append(p['Ia'])
                except:
                    continue

            if Ka_list:
                norm['Ka_mean'], norm['Ka_std'] = float(np.mean(Ka_list)), float(np.std(Ka_list))
                norm['Ca_mean'], norm['Ca_std'] = float(np.mean(Ca_list)), float(np.std(Ca_list))
                norm['Ia_mean'], norm['Ia_std'] = float(np.mean(Ia_list)), float(np.std(Ia_list))

        ka_mean, ka_std = norm.get('Ka_mean', 0.0), norm.get('Ka_std', 1.0)
        ca_mean, ca_std = norm.get('Ca_mean', 0.0), norm.get('Ca_std', 1.0)
        ia_mean, ia_std = norm.get('Ia_mean', 0.0), norm.get('Ia_std', 1.0)

        for folder in sorted(folders):
            fp = os.path.join(data_dir, folder)
            if not os.path.isdir(fp):
                continue

            try:
                p = json.load(open(os.path.join(fp, 'params.json')))
                ka_n = (p['Ka'] - ka_mean) / ka_std
                ca_n = (p['Ca'] - ca_mean) / ca_std
                ia_n = (p['Ia'] - ia_mean) / ia_std
            except:
                continue

            for fn in sorted(os.listdir(fp)):
                if not fn.endswith('_response.csv'):
                    continue

                try:
                    df = pd.read_csv(os.path.join(fp, fn))
                    U_val = float(fn.split('_')[0].lstrip('U'))
                    u_n = (U_val - u_mean) / u_std

                    alpha = df['Alpha'].values.astype(np.float32)
                    Cm = df['Cm'].values.astype(np.float32)

                    alpha_norm = (alpha - a_mean) / a_std
                    cm_norm = (Cm - cm_mean) / cm_std

                    L = len(alpha_norm)
                    total_len = input_len + output_len

                    for i in range(0, L - total_len, stride):
                        input_alpha = alpha_norm[i:i + input_len]
                        cond = np.array([ka_n, ca_n, ia_n, u_n], np.float32)
                        cond_seq = np.tile(cond, (input_len, 1))
                        input_seq = np.concatenate([
                            input_alpha.reshape(input_len, 1),
                            cond_seq
                        ], axis=1)

                        output_alpha = alpha_norm[i + input_len:i + total_len]


                        self.samples.append((input_seq, output_alpha, cond))
                except:
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        inp, out, cond = self.samples[idx]
        return (
            torch.tensor(inp, dtype=torch.float32),
            torch.tensor(out, dtype=torch.float32),
            torch.tensor(cond, dtype=torch.float32)
        )


class ImprovedTransformerSeq2Seq(nn.Module):
    def __init__(self, input_dim=5, d_model=128, nhead=8, num_encoder_layers=3,
                 num_decoder_layers=3, dim_feedforward=512, dropout=0.1,
                 input_len=16, output_len=8):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.d_model = d_model
        self.nhead = nhead


        self.input_projection = nn.Linear(input_dim, d_model)


        self.output_projection = nn.Linear(d_model, 1)


        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=5000)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)


        self.decoder_start_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.decoder_input_projection = nn.Linear(1, d_model)

    def forward(self, src, tgt=None, teacher_forcing_ratio=0.5):

        batch_size = src.size(0)


        src_emb = self.input_projection(src) * np.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        memory = self.transformer_encoder(src_emb)


        if tgt is not None and torch.rand(1).item() < teacher_forcing_ratio:


            start_token = self.decoder_start_token.repeat(batch_size, 1, 1)


            if self.output_len > 1:

                tgt_slice = tgt[:, :self.output_len - 1].unsqueeze(-1)

                tgt_slice_emb = self.decoder_input_projection(tgt_slice)

                tgt_emb = torch.cat([start_token, tgt_slice_emb], dim=1)
            else:

                tgt_emb = start_token
        else:

            tgt_emb = self.decoder_start_token.repeat(batch_size, self.output_len, 1)

        tgt_emb = self.pos_encoder(tgt_emb)


        tgt_mask = self._generate_square_subsequent_mask(self.output_len, device=src.device)


        decoder_output = self.transformer_decoder(
            tgt_emb, memory,
            tgt_mask=tgt_mask
        )


        output = self.output_projection(decoder_output).squeeze(-1)

        return output

    def _generate_square_subsequent_mask(self, sz, device):

        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].transpose(0, 1)
        return self.dropout(x)


def find_critical_points(alpha_seq, max_points=3):

    if torch.is_tensor(alpha_seq):
        if len(alpha_seq.shape) == 2:
            seq = alpha_seq[0].detach().cpu().numpy()
        else:
            seq = alpha_seq.detach().cpu().numpy()
    else:
        seq = alpha_seq

    n = len(seq)
    critical_points = []

    for i in range(1, n - 1):
        if (seq[i] > seq[i - 1] and seq[i] > seq[i + 1]) or                (seq[i] < seq[i - 1] and seq[i] < seq[i + 1]):
            critical_points.append(i)

    if not critical_points and n >= 2:
        critical_points = [0, n - 1]

    return sorted(critical_points[:max_points])


class AdaptiveToleranceManager:


    def __init__(self, base_epsilon=0.2, min_epsilon=0.05, max_epsilon=0.5,
                 adaptation_rate=0.05, warmup_steps=50,
                 schedule_type='cosine'):
        self.base_epsilon = base_epsilon
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.adaptation_rate = adaptation_rate
        self.warmup_steps = warmup_steps
        self.schedule_type = schedule_type
        self.step_count = 0

        self.residual_history = []
        self.epsilon_history = []
        self.system_state_history = []

    def get_epsilon(self, epoch=None, total_epochs=None, system_state=None, current_residual=None):
        self.step_count += 1

        if epoch is not None and total_epochs is not None:
            epsilon = self._get_scheduled_epsilon(epoch, total_epochs)
        else:
            epsilon = self.base_epsilon

        if system_state is not None:
            epsilon = self._adjust_by_system_state(epsilon, system_state)

        if current_residual is not None and len(self.residual_history) > 10:
            epsilon = self._adjust_by_residual_history(epsilon, current_residual)

        epsilon = max(self.min_epsilon, min(self.max_epsilon, epsilon))

        self.epsilon_history.append(epsilon)
        if current_residual is not None:
            self.residual_history.append(current_residual)

        return epsilon

    def _get_scheduled_epsilon(self, epoch, total_epochs):
        progress = epoch / total_epochs

        if self.schedule_type == 'cosine':
            epsilon = self.min_epsilon + 0.5 * (self.max_epsilon - self.min_epsilon) *                      (1 + np.cos(np.pi * progress))
        elif self.schedule_type == 'linear':
            epsilon = self.max_epsilon - (self.max_epsilon - self.min_epsilon) * progress
        elif self.schedule_type == 'step':
            if progress < 0.3:
                epsilon = self.max_epsilon
            elif progress < 0.7:
                epsilon = self.max_epsilon * 0.7
            else:
                epsilon = self.max_epsilon * 0.3
        else:
            epsilon = self.base_epsilon

        return epsilon

    def _adjust_by_system_state(self, epsilon, system_state):
        gradient_multiplier = 1.0
        curvature_multiplier = 1.0

        if 'gradient' in system_state:
            gradient = abs(system_state['gradient'])
            if gradient > 0.8:
                gradient_multiplier = 1.5
            elif gradient > 0.5:
                gradient_multiplier = 1.2
            elif gradient < 0.1:
                gradient_multiplier = 0.8

        if 'curvature' in system_state:
            curvature = abs(system_state['curvature'])
            if curvature > 0.5:
                curvature_multiplier = 1.3
            elif curvature > 0.2:
                curvature_multiplier = 1.1
            elif curvature < 0.05:
                curvature_multiplier = 0.9

        return epsilon * gradient_multiplier * curvature_multiplier

    def _adjust_by_residual_history(self, epsilon, current_residual):
        if len(self.residual_history) == 0:
            return epsilon

        recent_residuals = self.residual_history[-min(100, len(self.residual_history)):]
        mean_residual = np.mean(np.abs(recent_residuals))
        std_residual = np.std(recent_residuals)

        if abs(current_residual) > mean_residual + 3 * std_residual:
            epsilon = epsilon * 1.5
        elif abs(current_residual) > mean_residual + 2 * std_residual:
            epsilon = epsilon * 1.2
        elif abs(current_residual) < mean_residual - std_residual:
            epsilon = epsilon * 0.9

        return epsilon

    def get_statistics(self):
        if not self.epsilon_history:
            return {}

        return {
            'mean_epsilon': float(np.mean(self.epsilon_history)),
            'std_epsilon': float(np.std(self.epsilon_history)),
            'min_epsilon': float(np.min(self.epsilon_history)),
            'max_epsilon': float(np.max(self.epsilon_history)),
            'adaptation_steps': self.step_count,
            'schedule_type': self.schedule_type
        }


class EnhancedPhysicsConstraintEvaluator:


    def __init__(self, norm, base_model, base_model_seq_len, dt=0.005):
        self.norm = norm
        self.base_model = base_model
        self.base_model_seq_len = base_model_seq_len
        self.dt = dt

        self.c, self.l, self.rho = 0.156, 0.61, 1.225

        self.a_mean, self.a_std = norm['a_mean'], norm['a_std']
        self.cm_mean, self.cm_std = norm['cm_mean'], norm['cm_std']
        self.u_mean, self.u_std = norm['u_mean'], norm['u_std']
        self.Ka_mean, self.Ka_std = norm.get('Ka_mean', 0.0), norm.get('Ka_std', 1.0)
        self.Ca_mean, self.Ca_std = norm.get('Ca_mean', 0.0), norm.get('Ca_std', 1.0)
        self.Ia_mean, self.Ia_std = norm.get('Ia_mean', 0.0), norm.get('Ia_std', 1.0)

        self.tolerance_manager = AdaptiveToleranceManager(
            base_epsilon=0.2, min_epsilon=0.05, max_epsilon=0.5,
            schedule_type='cosine'
        )

        self.reset_statistics()

    def identify_region(self, alpha_history, alpha_current):
        if len(alpha_history) < 5:
            return 'normal_region'

        if torch.is_tensor(alpha_current):
            alpha_current = alpha_current.item() if alpha_current.numel() == 1 else float(alpha_current)

        recent_alphas = alpha_history[-5:] + [alpha_current]
        gradients = np.diff(recent_alphas) / self.dt

        if len(gradients) >= 3:
            curvature = np.diff(gradients) / self.dt
            max_curvature = np.max(np.abs(curvature))
        else:
            max_curvature = 0

        max_gradient = np.max(np.abs(gradients))

        if max_gradient < 0.05 and max_curvature < 0.02:
            return 'normal_region'
        elif max_gradient > 0.3 or max_curvature > 0.15:
            return 'extreme_region'
        else:
            return 'critical_region'

    def calculate_physics_residuals(self, alpha_pred_sequence, src_sequence,
                                    return_detailed=False, current_epsilon=None):
        batch_size, output_len = alpha_pred_sequence.shape
        device = alpha_pred_sequence.device


        Ka_norm = src_sequence[:, -1, 1]
        Ca_norm = src_sequence[:, -1, 2]
        Ia_norm = src_sequence[:, -1, 3]
        U_norm = src_sequence[:, -1, 4]


        Ka = Ka_norm * self.Ka_std + self.Ka_mean
        Ca = Ca_norm * self.Ca_std + self.Ca_mean
        Ia = Ia_norm * self.Ia_std + self.Ia_mean
        U = U_norm * self.u_std + self.u_mean

        alpha_pred_real = alpha_pred_sequence * self.a_std + self.a_mean

        alpha_hist = src_sequence[:, :, 0]
        alpha_hist_real = alpha_hist * self.a_std + self.a_mean

        all_residuals = []
        all_relative_residuals = []
        detailed_stats = {
            'residuals_by_time': [],
            'relative_residuals_by_time': [],
            'terms_analysis': [],
            'regions_by_time': [],
            'epsilon_by_time': []
        }


        region_counts = {region: 0 for region in self.stats['region_stats'].keys()}
        region_violations = {region: 0 for region in self.stats['region_stats'].keys()}
        region_residual_sums = {region: 0.0 for region in self.stats['region_stats'].keys()}

        for t_idx in range(output_len):
            alpha_t_pred = alpha_pred_real[:, t_idx]

            if t_idx == 0:
                alpha_t1 = alpha_hist_real[:, -1]
                alpha_t2 = alpha_hist_real[:, -2]
            elif t_idx == 1:
                alpha_t1 = alpha_pred_real[:, 0]
                alpha_t2 = alpha_hist_real[:, -1]
            else:
                alpha_t1 = alpha_pred_real[:, t_idx - 1]
                alpha_t2 = alpha_pred_real[:, t_idx - 2]

            dt_safe = self.dt + 1e-8
            α̇ = (alpha_t_pred - alpha_t1) / dt_safe
            α̈ = (alpha_t_pred - 2 * alpha_t1 + alpha_t2) / (dt_safe ** 2 + 1e-8)


            regions = []
            for b in range(batch_size):
                if t_idx == 0:
                    alpha_hist_numpy = alpha_hist_real[b].detach().cpu().numpy().tolist()
                else:
                    alpha_hist_numpy = alpha_pred_real[b, :t_idx].detach().cpu().numpy().tolist()

                alpha_current_scalar = alpha_t_pred[b].item()
                region = self.identify_region(alpha_hist_numpy, alpha_current_scalar)
                regions.append(region)
                region_counts[region] += 1

            system_state = {
                'gradient': α̇[0].item() if batch_size > 0 else 0,
                'curvature': α̈[0].item() if batch_size > 0 else 0
            }

            hist_for_cm_list = []
            for b in range(batch_size):
                if t_idx == 0:
                    hist = alpha_hist[b, -self.base_model_seq_len:]
                else:
                    available_hist = torch.cat([
                        alpha_hist[b, :],
                        alpha_pred_sequence[b, :t_idx]
                    ])
                    if len(available_hist) >= self.base_model_seq_len:
                        hist = available_hist[-self.base_model_seq_len:]
                    else:
                        padding = torch.full((self.base_model_seq_len - len(available_hist),),
                                             available_hist[0], device=device)
                        hist = torch.cat([padding, available_hist])

                hist_for_cm_list.append(hist.unsqueeze(0))

            hist_for_cm = torch.cat(hist_for_cm_list, dim=0)
            hist_for_cm = hist_for_cm.unsqueeze(-1)


            u_val = U_norm.unsqueeze(-1).unsqueeze(-1)

            with torch.no_grad():
                try:
                    cm_pred_norm, _ = self.base_model(hist_for_cm, u_val)
                    cm_pred_real = cm_pred_norm.squeeze(-1) * self.cm_std + self.cm_mean
                except Exception as e:
                    cm_pred_real = torch.zeros(batch_size, 1, device=device)

            inertia_term = Ia * α̈
            damping_term = Ca * α̇
            stiffness_term = Ka * alpha_t_pred
            aerodynamic_term = 0.5 * self.rho * (U ** 2 + 1e-6) * (self.c ** 2) * self.l * cm_pred_real.squeeze(-1)

            resid = inertia_term + damping_term + stiffness_term - aerodynamic_term

            M0 = 0.5 * self.rho * (U ** 2 + 1e-3) * (self.c ** 2) * self.l
            normalized_resid = resid / M0.detach()

            term_max = torch.max(torch.stack([
                torch.abs(inertia_term),
                torch.abs(damping_term),
                torch.abs(stiffness_term),
                torch.abs(aerodynamic_term)
            ], dim=0), dim=0)[0]
            relative_resid = resid / (term_max.detach() + 1e-6)

            if current_epsilon is not None:
                epsilon = current_epsilon
            else:
                epsilon = self.tolerance_manager.get_epsilon(
                    system_state=system_state,
                    current_residual=normalized_resid[0].item() if batch_size > 0 else 0
                )

            all_residuals.append(normalized_resid)
            all_relative_residuals.append(relative_resid)


            violation_mask = (torch.abs(normalized_resid) > epsilon)
            for b in range(batch_size):
                region = regions[b]
                if violation_mask[b].item():
                    region_violations[region] += 1
                region_residual_sums[region] += normalized_resid[b].abs().item()

            if return_detailed:
                detailed_stats['residuals_by_time'].append(normalized_resid.detach().cpu().numpy())
                detailed_stats['relative_residuals_by_time'].append(relative_resid.detach().cpu().numpy())
                detailed_stats['regions_by_time'].append(regions)
                detailed_stats['epsilon_by_time'].append(epsilon)
                detailed_stats['terms_analysis'].append({
                    'inertia': inertia_term.detach().cpu().numpy(),
                    'damping': damping_term.detach().cpu().numpy(),
                    'stiffness': stiffness_term.detach().cpu().numpy(),
                    'aerodynamic': aerodynamic_term.detach().cpu().numpy()
                })

        all_residuals_tensor = torch.stack(all_residuals, dim=1)
        all_relative_residuals_tensor = torch.stack(all_relative_residuals, dim=1)

        self.stats['total_samples'] += batch_size * output_len
        self.stats['epsilon_history'].extend(detailed_stats['epsilon_by_time'] if return_detailed else [])

        residuals_flat = all_residuals_tensor.detach().cpu().numpy().flatten()
        relative_residuals_flat = all_relative_residuals_tensor.detach().cpu().numpy().flatten()

        self.stats['residual_history'].extend(residuals_flat)
        self.stats['relative_residuals'].extend(relative_residuals_flat)
        self.stats['per_time_step_residuals'].append(all_residuals_tensor.detach().cpu().numpy())


        for region in self.stats['region_stats'].keys():
            self.stats['region_stats'][region]['count'] += region_counts[region]
            self.stats['region_stats'][region]['violations'] += region_violations[region]
            self.stats['region_stats'][region]['mean_residual'] += region_residual_sums[region]

        if return_detailed:
            return all_residuals_tensor, detailed_stats
        else:
            return all_residuals_tensor

    def get_summary_statistics(self):
        if len(self.stats['residual_history']) == 0:
            return None

        residuals = np.array(self.stats['residual_history'])
        relative_residuals = np.array(self.stats['relative_residuals'])

        if residuals.ndim > 1:
            residuals = residuals.flatten()
        if relative_residuals.ndim > 1:
            relative_residuals = relative_residuals.flatten()

        region_stats = {}
        for region, stats_data in self.stats['region_stats'].items():
            if stats_data['count'] > 0:
                region_stats[region] = {
                    'count': stats_data['count'],
                    'violation_rate': stats_data['violations'] / stats_data['count'] if stats_data['count'] > 0 else 0,
                    'mean_residual': stats_data['mean_residual'] / stats_data['count'] if stats_data['count'] > 0 else 0
                }

        stats_summary = {
            'total_samples': self.stats['total_samples'],
            'violation_count': self.stats['physics_violations'],
            'violation_rate': self.stats['physics_violations'] / self.stats['total_samples'] if self.stats[
                                                                                                    'total_samples'] > 0 else 0,
            'mean_residual': float(np.mean(np.abs(residuals))),
            'std_residual': float(np.std(residuals)),
            'max_residual': float(np.max(np.abs(residuals))),
            'mean_relative_residual': float(np.mean(np.abs(relative_residuals))),
            'std_relative_residual': float(np.std(relative_residuals)),
            'skewness': float(stats.skew(residuals)) if len(residuals) > 0 else 0.0,
            'kurtosis': float(stats.kurtosis(residuals)) if len(residuals) > 0 else 0.0,
            'region_stats': region_stats,
            'epsilon_stats': self.tolerance_manager.get_statistics()
        }

        return stats_summary

    def reset_statistics(self):
        self.stats = {
            'total_samples': 0,
            'physics_violations': 0,
            'residual_history': [],
            'violation_history': [],
            'per_time_step_residuals': [],
            'region_stats': {
                'normal_region': {'count': 0, 'violations': 0, 'mean_residual': 0},
                'critical_region': {'count': 0, 'violations': 0, 'mean_residual': 0},
                'extreme_region': {'count': 0, 'violations': 0, 'mean_residual': 0}
            },
            'epsilon_history': [],
            'relative_residuals': []
        }

        self.tolerance_manager = AdaptiveToleranceManager(
            base_epsilon=0.2, min_epsilon=0.05, max_epsilon=0.5
        )


def calculate_physics_constraint_efficient(alpha_pred_sequence, src_sequence, base_model, norm,
                                           base_model_seq_len, dt=0.005, constraint_config=None,
                                           evaluator=None, current_epsilon=None, epoch=None, total_epochs=None):
    if constraint_config is None:
        constraint_config = {
            'mode': 'hybrid',
            'num_points': 4,
            'use_critical_points': True,
            'adaptive_weighting': True,
            'tolerance_params': {
                'epsilon': 0.2,
                'loss_type': 'huber'
            }
        }

    batch_size, output_len = alpha_pred_sequence.shape
    device = alpha_pred_sequence.device

    c, l, rho = 0.156, 0.61, 1.225

    a_mean, a_std = norm['a_mean'], norm['a_std']
    cm_mean, cm_std = norm['cm_mean'], norm['cm_std']
    u_mean, u_std = norm['u_mean'], norm['u_std']
    Ka_mean, Ka_std = norm.get('Ka_mean', 0.0), norm.get('Ka_std', 1.0)
    Ca_mean, Ca_std = norm.get('Ca_mean', 0.0), norm.get('Ca_std', 1.0)
    Ia_mean, Ia_std = norm.get('Ia_mean', 0.0), norm.get('Ia_std', 1.0)


    Ka_norm = src_sequence[:, -1, 1]
    Ca_norm = src_sequence[:, -1, 2]
    Ia_norm = src_sequence[:, -1, 3]
    U_norm = src_sequence[:, -1, 4]


    Ka = Ka_norm * Ka_std + Ka_mean
    Ca = Ca_norm * Ca_std + Ca_mean
    Ia = Ia_norm * Ia_std + Ia_mean
    U = U_norm * u_std + u_mean

    alpha_pred_real = alpha_pred_sequence * a_std + a_mean

    alpha_hist = src_sequence[:, :, 0]
    alpha_hist_real = alpha_hist * a_std + a_mean

    mode = constraint_config.get('mode', 'hybrid')

    if mode == 'single':
        selected_indices = [0]
    elif mode == 'multi':
        num_points = min(constraint_config.get('num_points', 4), output_len)
        if num_points >= output_len:
            selected_indices = list(range(output_len))
        else:
            selected_indices = np.linspace(0, output_len - 1, num_points, dtype=int).tolist()
    elif mode == 'critical':
        selected_indices = find_critical_points(alpha_pred_real.detach(),
                                                max_points=constraint_config.get('num_points', 3))
    else:
        critical_indices = find_critical_points(alpha_pred_real.detach(), max_points=2)
        num_points = constraint_config.get('num_points', 4)
        if len(critical_indices) >= num_points:
            selected_indices = critical_indices[:num_points]
        else:
            remaining_points = num_points - len(critical_indices)
            uniform_indices = list(range(0, output_len, max(1, output_len // remaining_points)))
            selected_indices = sorted(set(critical_indices + uniform_indices[:remaining_points]))

    total_loss = torch.tensor(0.0, device=device)
    total_weight = 0

    tolerance_params = constraint_config.get('tolerance_params', {})
    if current_epsilon is None:
        epsilon = tolerance_params.get('epsilon', 0.2)
    else:
        epsilon = current_epsilon

    loss_type = tolerance_params.get('loss_type', 'balanced')
    epsilon_tensor = torch.tensor(epsilon, device=device, dtype=alpha_pred_sequence.dtype)

    region_weights = {
        'normal_region': 0.8,
        'critical_region': 1.2,
        'extreme_region': 0.6
    }


    if evaluator is not None:
        region_counts = {region: 0 for region in evaluator.stats['region_stats'].keys()}
        region_violations = {region: 0 for region in evaluator.stats['region_stats'].keys()}

    for t_idx in selected_indices:
        if t_idx >= output_len:
            continue

        alpha_t_pred = alpha_pred_real[:, t_idx]

        if t_idx == 0:
            alpha_t1 = alpha_hist_real[:, -1]
            alpha_t2 = alpha_hist_real[:, -2]
        elif t_idx == 1:
            alpha_t1 = alpha_pred_real[:, 0]
            alpha_t2 = alpha_hist_real[:, -1]
        else:
            alpha_t1 = alpha_pred_real[:, t_idx - 1]
            alpha_t2 = alpha_pred_real[:, t_idx - 2]

        dt_safe = dt + 1e-8
        α̇ = (alpha_t_pred - alpha_t1) / dt_safe
        α̈ = (alpha_t_pred - 2 * alpha_t1 + alpha_t2) / (dt_safe ** 2 + 1e-8)


        regions = []
        if evaluator is not None and constraint_config.get('use_region_weights', False):
            for b in range(batch_size):
                if t_idx == 0:
                    alpha_hist_numpy = alpha_hist_real[b].detach().cpu().numpy().tolist()
                else:
                    alpha_hist_numpy = alpha_pred_real[b, :t_idx].detach().cpu().numpy().tolist()

                alpha_current_scalar = alpha_t_pred[b].item()
                region = evaluator.identify_region(alpha_hist_numpy, alpha_current_scalar)
                regions.append(region)
                region_counts[region] += 1

        hist_for_cm_list = []
        for b in range(batch_size):
            if t_idx == 0:
                hist = alpha_hist[b, -base_model_seq_len:]
            else:
                available_hist = torch.cat([
                    alpha_hist[b, :],
                    alpha_pred_sequence[b, :t_idx]
                ])
                if len(available_hist) >= base_model_seq_len:
                    hist = available_hist[-base_model_seq_len:]
                else:
                    padding = torch.full((base_model_seq_len - len(available_hist),),
                                         available_hist[0], device=device)
                    hist = torch.cat([padding, available_hist])

            hist_for_cm_list.append(hist.unsqueeze(0))

        hist_for_cm = torch.cat(hist_for_cm_list, dim=0)
        hist_for_cm = hist_for_cm.unsqueeze(-1)


        u_val = U_norm.unsqueeze(-1).unsqueeze(-1)

        with torch.no_grad():
            try:
                cm_pred_norm, _ = base_model(hist_for_cm, u_val)
                cm_pred_real = cm_pred_norm.squeeze(-1) * cm_std + cm_mean
            except Exception as e:
                cm_pred_real = torch.zeros(batch_size, 1, device=device)

        M = 0.5 * rho * (U ** 2 + 1e-6) * (c ** 2) * l * cm_pred_real.squeeze(-1)
        resid = Ia * α̈ + Ca * α̇ + Ka * alpha_t_pred - M

        M0 = 0.5 * rho * (U ** 2 + 1e-3) * (c ** 2) * l
        resid_normalized = resid / M0.detach()

        term_abs = torch.stack([
            torch.abs(Ia * α̈),
            torch.abs(Ca * α̇),
            torch.abs(Ka * alpha_t_pred),
            torch.abs(M)
        ], dim=0)
        term_max = torch.max(term_abs, dim=0)[0]
        resid_relative = resid / (term_max.detach() + 1e-6)

        resid_for_loss = resid_relative

        if evaluator is not None:
            resid_flat = resid_normalized.detach().cpu().numpy().flatten()
            evaluator.stats['residual_history'].extend(resid_flat)

            resid_rel_flat = resid_relative.detach().cpu().numpy().flatten()
            evaluator.stats['relative_residuals'].extend(resid_rel_flat)

            violation_mask = (torch.abs(resid_normalized) > epsilon)
            evaluator.stats['physics_violations'] += violation_mask.sum().item()
            evaluator.stats['total_samples'] += batch_size


            if len(regions) == batch_size:
                for b in range(batch_size):
                    region = regions[b]
                    if violation_mask[b].item():
                        region_violations[region] += 1

        scale_factor = tolerance_params.get('scale_factor', 0.05)

        region_weight = 1.0
        if constraint_config.get('use_region_weights', False) and evaluator is not None and len(regions) == batch_size:

            region_weight = region_weights.get(regions[0], 1.0)

        if constraint_config.get('adaptive_weighting', True):
            time_weight = 1.0 / (t_idx + 1)
        else:
            time_weight = 1.0

        weight = time_weight * region_weight

        if loss_type == 'mse':
            r = torch.relu(torch.abs(resid_for_loss) - epsilon_tensor)
            loss_step = (r ** 2).mean()
        elif loss_type == 'huber':
            abs_resid = torch.abs(resid_for_loss)
            loss_step = torch.where(
                abs_resid <= epsilon_tensor,
                0.5 * (abs_resid ** 2),
                epsilon_tensor * (abs_resid - 0.5 * epsilon_tensor)
            ).mean()
        elif loss_type == 'balanced':
            abs_resid = torch.abs(resid_for_loss)

            small_mask = abs_resid <= epsilon_tensor
            medium_mask = (abs_resid > epsilon_tensor) & (abs_resid <= 2 * epsilon_tensor)
            large_mask = abs_resid > 2 * epsilon_tensor

            loss_small = (abs_resid[small_mask] ** 2).sum() if small_mask.any() else 0
            loss_medium = (epsilon_tensor * (
                    abs_resid[medium_mask] - 0.5 * epsilon_tensor)).sum() if medium_mask.any() else 0
            loss_large = (2 * epsilon_tensor * (
                    abs_resid[large_mask] - epsilon_tensor) + epsilon_tensor ** 2).sum() if large_mask.any() else 0

            total_elements = abs_resid.numel()
            loss_step = (
                                loss_small + loss_medium + loss_large) / total_elements if total_elements > 0 else torch.tensor(
                0.0, device=device)
        else:
            r = torch.relu(torch.abs(resid_for_loss) - epsilon_tensor)
            loss_step = (r ** 2).mean()

        total_loss += weight * loss_step * scale_factor
        total_weight += weight


    if evaluator is not None and 'region_counts' in locals():
        for region in evaluator.stats['region_stats'].keys():
            evaluator.stats['region_stats'][region]['count'] += region_counts.get(region, 0)
            evaluator.stats['region_stats'][region]['violations'] += region_violations.get(region, 0)

    if total_weight > 0:
        return total_loss / total_weight
    else:
        return torch.tensor(0.0, device=device)


def train_physics_constrained_model_custom(data_dir, norm, base_model, base_model_seq_len,
                                           input_len=16, output_len=8, stride=4,
                                           batch_size=128, epochs=30, lr=5e-4, patience=10,
                                           train_folders=None, val_folders=None, test_folders=None,
                                           device='cuda', seed=42):

    set_seed(seed)

    print("\n" + "=" * 60)
    print("🚀 Training with custom dataset split")
    print("=" * 60)


    if train_folders is None or val_folders is None:
        raise ValueError("train_folders and val_folders must be provided")

    print(f"📊 Using custom dataset split:")
    print(f"  Training set: {len(train_folders)} folders")
    print(f"  Validation set: {len(val_folders)} folders")
    if test_folders:
        print(f"  Test set: {len(test_folders)} folders")


    train_ds = Seq2SeqAlphaDataset(data_dir, norm, train_folders, input_len, output_len, stride,
                                   calc_norm=True, teacher_forcing_ratio=0.5)

    val_ds = Seq2SeqAlphaDataset(data_dir, norm, val_folders, input_len, output_len, stride,
                                 calc_norm=False, external_norm=norm, teacher_forcing_ratio=0.0)


    test_ds = None
    if test_folders:
        test_ds = Seq2SeqAlphaDataset(data_dir, norm, test_folders, input_len, output_len, stride,
                                      calc_norm=False, external_norm=norm, teacher_forcing_ratio=0.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"📈 Data loading completed:")
    print(f"  Training samples: {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    if test_ds:
        print(f"  Test samples: {len(test_ds)}")


    model = ImprovedTransformerSeq2Seq(
        input_dim=5, d_model=128, nhead=8,
        num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=512, dropout=0.0,
        input_len=input_len, output_len=output_len
    ).to(device)

    print(f"🤖 Improved Transformer model with Teacher Forcing support")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")


    constraint_config = {
        'constraint_mode': 'both',
        'λ_max': 0.1,
        'warm_epochs_ratio': 0.4,
        'teacher_forcing_start': 0.8,
        'teacher_forcing_end': 0.1,
        'tolerance_params': {
            'epsilon': 0.2,
            'adaptive_epsilon': True,
            'epsilon_schedule': 'cosine',
            'initial_epsilon': 0.4,
            'final_epsilon': 0.08,
            'min_epsilon': 0.02,
            'max_epsilon': 0.5,
            'loss_type': 'balanced',
            'clip_grad': True,
            'scale_factor': 0.01
        },
        'physics_config': {
            'mode': 'hybrid',
            'num_points': 4,
            'use_critical_points': True,
            'adaptive_weighting': True,
            'use_region_weights': True
        }
    }


    model, train_history, physics_evaluator = train_with_physics_evaluation(
        model=model,
        base_model=base_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        norm=norm,
        base_model_seq_len=base_model_seq_len,
        epochs=epochs,
        lr=lr,
        patience=patience,
        model_path='best_physics_constrained_model_tf.pt',
        constraint_mode=constraint_config['constraint_mode'],
        λ_max=constraint_config['λ_max'],
        warm_epochs_ratio=constraint_config['warm_epochs_ratio'],
        tolerance_params=constraint_config['tolerance_params'],
        physics_config=constraint_config['physics_config'],
        save_dir="physics_constrained_training_tf",
        teacher_forcing_start=constraint_config['teacher_forcing_start'],
        teacher_forcing_end=constraint_config['teacher_forcing_end']
    )

    return model, train_history, test_folders, output_len, physics_evaluator


def train_with_physics_evaluation(model, base_model, train_loader, val_loader, device, norm,
                                  base_model_seq_len, epochs=100, lr=1e-3, patience=20,
                                  model_path='best_transformer_seq2seq.pt',
                                  constraint_mode="both", λ_max=0.8, warm_epochs_ratio=0.4,
                                  tolerance_params=None, physics_config=None,
                                  save_dir="training_with_physics_eval", teacher_forcing_start=1.0,
                                  teacher_forcing_end=0.1):

    os.makedirs(save_dir, exist_ok=True)

    if tolerance_params is None:
        tolerance_params = {
            'epsilon': 0.2,
            'adaptive_epsilon': True,
            'epsilon_schedule': 'cosine',
            'initial_epsilon': 0.4,
            'final_epsilon': 0.08,
            'min_epsilon': 0.02,
            'max_epsilon': 0.5,
            'loss_type': 'balanced',
            'clip_grad': True,
            'scale_factor': 0.05
        }

    if physics_config is None:
        physics_config = {
            'mode': 'hybrid',
            'num_points': 4,
            'use_critical_points': True,
            'adaptive_weighting': True,
            'use_region_weights': True,
            'tolerance_params': tolerance_params
        }

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.MSELoss()

    physics_evaluator = EnhancedPhysicsConstraintEvaluator(norm, base_model, base_model_seq_len)

    best_val_loss = float('inf')
    patience_counter = 0
    train_history = {
        'train_loss': [], 'val_loss': [], 'learning_rate': [],
        'data_loss': [], 'phy_loss': [], 'epsilon': [],
        'constraint_points': [], 'lambda_phy': [],
        'physics_stats': [], 'teacher_forcing_ratio': []
    }

    warm_epochs = int(warm_epochs_ratio * epochs)

    for epoch in range(1, epochs + 1):

        model.train()
        train_loss = 0.0
        data_loss_total = 0.0
        phy_loss_total = 0.0

        if constraint_mode == "both":
            if epoch <= warm_epochs:
                λ_phy = λ_max * (epoch / warm_epochs) ** 2
            else:
                λ_phy = λ_max
        else:
            λ_phy = 0.0


        teacher_forcing_ratio = teacher_forcing_start - (teacher_forcing_start - teacher_forcing_end) * (
                epoch / epochs)
        teacher_forcing_ratio = max(teacher_forcing_end, teacher_forcing_ratio)


        if tolerance_params.get('adaptive_epsilon', True):
            if tolerance_params.get('epsilon_schedule', 'cosine') == 'cosine':
                progress = epoch / epochs
                current_epsilon = tolerance_params.get('final_epsilon', 0.08) +                                  0.5 * (tolerance_params.get('initial_epsilon', 0.4) -
                                         tolerance_params.get('final_epsilon', 0.08)) *                                  (1 + np.cos(np.pi * progress))
            else:
                progress = epoch / epochs
                current_epsilon = tolerance_params.get('initial_epsilon', 0.4) -                                  (tolerance_params.get('initial_epsilon', 0.4) -
                                   tolerance_params.get('final_epsilon', 0.08)) * progress

            current_epsilon = max(tolerance_params.get('min_epsilon', 0.02),
                                  min(tolerance_params.get('max_epsilon', 0.5), current_epsilon))
        else:
            current_epsilon = tolerance_params.get('epsilon', 0.2)


        current_physics_config = copy.deepcopy(physics_config)
        if 'tolerance_params' in current_physics_config:
            current_physics_config['tolerance_params']['epsilon'] = current_epsilon
        else:
            current_physics_config['tolerance_params'] = {'epsilon': current_epsilon}

        if epoch < warm_epochs * 0.3:

            epoch_physics_config = copy.deepcopy(current_physics_config)
            epoch_physics_config['num_points'] = max(1, current_physics_config.get('num_points', 4) // 4)
            epoch_physics_config['mode'] = 'critical'
        elif epoch < warm_epochs * 0.6:
            epoch_physics_config = copy.deepcopy(current_physics_config)
            epoch_physics_config['num_points'] = max(2, current_physics_config.get('num_points', 4) // 2)
            epoch_physics_config['mode'] = 'hybrid'
        else:
            epoch_physics_config = copy.deepcopy(current_physics_config)

        constraint_points = epoch_physics_config.get('num_points', 4)
        if epoch_physics_config.get('mode') == 'single':
            constraint_points = 1

        train_history['constraint_points'].append(constraint_points)
        train_history['lambda_phy'].append(λ_phy)
        train_history['teacher_forcing_ratio'].append(teacher_forcing_ratio)

        physics_evaluator.reset_statistics()

        train_bar = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs} [Train]',
                         leave=False, ncols=100)

        for batch_idx, (src, tgt, cond) in enumerate(train_bar):
            src, tgt = src.to(device), tgt.to(device)

            optimizer.zero_grad()


            output = model(src, tgt=tgt, teacher_forcing_ratio=teacher_forcing_ratio)

            loss_data = criterion(output, tgt)
            data_loss_total += loss_data.item()

            loss_phy = torch.tensor(0.0, device=device)
            if constraint_mode == "both" and λ_phy > 0 and output.shape[1] > 0:
                try:
                    loss_phy = calculate_physics_constraint_efficient(
                        output, src, base_model, norm, base_model_seq_len,
                        dt=0.005, constraint_config=epoch_physics_config,
                        evaluator=physics_evaluator,
                        current_epsilon=current_epsilon,
                        epoch=epoch,
                        total_epochs=epochs
                    )
                    phy_loss_total += loss_phy.item()
                except Exception as e:
                    print(f"Physics constraint calculation failed (Epoch {epoch}, Batch {batch_idx}): {e}")
                    loss_phy = torch.tensor(0.0, device=device)

            if constraint_mode == "both":
                loss = loss_data + λ_phy * loss_phy
            else:
                loss = loss_data

            loss.backward()

            if tolerance_params.get('clip_grad', True):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

            train_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Data': f'{loss_data.item():.4f}',
                'Phy': f'{loss_phy.item():.4f}' if constraint_mode == "both" else 'N/A',
                'λ': f'{λ_phy:.3f}' if constraint_mode == "both" else 'N/A',
                'ε': f'{current_epsilon:.3f}',
                'TF': f'{teacher_forcing_ratio:.3f}'
            })

        train_bar.close()

        avg_train_loss = train_loss / len(train_loader)
        avg_data_loss = data_loss_total / len(train_loader)
        avg_phy_loss = phy_loss_total / len(train_loader) if constraint_mode == "both" else 0.0


        model.eval()
        val_loss = 0.0

        val_bar = tqdm(val_loader, desc=f'Epoch {epoch}/{epochs} [Val]',
                       leave=False, ncols=100)

        with torch.no_grad():
            for src, tgt, cond in val_bar:
                src, tgt = src.to(device), tgt.to(device)


                output = model(src, tgt=None, teacher_forcing_ratio=0.0)


                loss = criterion(output, tgt)
                val_loss += loss.item()

                val_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

        val_bar.close()

        avg_val_loss = val_loss / len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        physics_stats = physics_evaluator.get_summary_statistics()
        if physics_stats:
            train_history['physics_stats'].append(physics_stats)

        train_history['train_loss'].append(avg_train_loss)
        train_history['val_loss'].append(avg_val_loss)
        train_history['learning_rate'].append(current_lr)
        train_history['data_loss'].append(avg_data_loss)
        train_history['phy_loss'].append(avg_phy_loss)
        train_history['epsilon'].append(current_epsilon)

        if constraint_mode == "both":
            if physics_stats:
                print(f'Epoch {epoch:03d}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, '
                      f'Data Loss: {avg_data_loss:.6f}, Phy Loss: {avg_phy_loss:.6f}, '
                      f'LR: {current_lr:.2e}, λ_phy: {λ_phy:.3f}, ε: {current_epsilon:.3f}, '
                      f'TF: {teacher_forcing_ratio:.3f}, Points: {constraint_points}, '
                      f'Phy Violation: {physics_stats["violation_rate"]:.4f}, '
                      f'Mean Rel Res: {physics_stats["mean_relative_residual"]:.6f}')
            else:
                print(f'Epoch {epoch:03d}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, '
                      f'Data Loss: {avg_data_loss:.6f}, Phy Loss: {avg_phy_loss:.6f}, '
                      f'LR: {current_lr:.2e}, λ_phy: {λ_phy:.3f}, ε: {current_epsilon:.3f}, '
                      f'TF: {teacher_forcing_ratio:.3f}, Points: {constraint_points}')
        else:
            print(
                f'Epoch {epoch:03d}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, '
                f'LR: {current_lr:.2e}, TF: {teacher_forcing_ratio:.3f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
            print(f'✅ Saved best model with val loss: {best_val_loss:.6f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch}')
                break

        if epoch % 3 == 0 and constraint_mode == "both" and physics_stats:
            epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch:03d}")
            os.makedirs(epoch_save_dir, exist_ok=True)


    try:
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    except:
        model.load_state_dict(torch.load(model_path, map_location=device))

    if constraint_mode == "both":
        final_physics_dir = os.path.join(save_dir, "final_physics_analysis")
        os.makedirs(final_physics_dir, exist_ok=True)


        final_stats = physics_evaluator.get_summary_statistics()
    else:
        final_stats = None

    return model, train_history, physics_evaluator


def train_physics_constrained_model(data_dir, norm, base_model, base_model_seq_len,
                                    input_len=16, output_len=8, stride=4,
                                    batch_size=128, epochs=30, lr=5e-4, patience=10,
                                    train_ratio=0.7, val_ratio=0.15, device='cuda', seed=42):

    set_seed(seed)

    print("\n" + "=" * 60)
    print("🚀 Starting training of physics-constrained model with Teacher Forcing")
    print("=" * 60)


    all_folders = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    random.shuffle(all_folders)
    n_total = len(all_folders)

    print(f"📁 Total folders: {n_total}")


    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    n_test = n_total - n_train - n_val

    train_folders = all_folders[:n_train]
    val_folders = all_folders[n_train:n_train + n_val]
    test_folders = all_folders[n_train + n_val:]


    split_info = {
        'train_folders': train_folders,
        'val_folders': val_folders,
        'test_folders': test_folders,
        'all_folders': all_folders,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': 1.0 - train_ratio - val_ratio,
        'n_total': n_total,
        'n_train': n_train,
        'n_val': n_val,
        'n_test': n_test,
        'seed': seed,
        'shuffled': True
    }

    split_info_path = os.path.join(data_dir, "dataset_split_info.json")
    with open(split_info_path, 'w') as f:
        json.dump(split_info, f, indent=2)

    print(f"💾 Dataset split info saved to {split_info_path}")
    print(f"📊 Dataset split (random shuffle with seed={seed}):")
    print(f"  Training set: {len(train_folders)} folders ({train_ratio:.0%})")
    print(f"  Validation set: {len(val_folders)} folders ({val_ratio:.0%})")
    print(f"  Test set: {len(test_folders)} folders (remaining)")
    print(f"  Test folders: {test_folders}")


    if len(test_folders) == 0:
        print("⚠️ Warning: No test folders, using validation set as test set")
        test_folders = val_folders
        val_folders = train_folders[-5:] if len(train_folders) >= 10 else train_folders[-2:]
        train_folders = train_folders[:-len(val_folders)]


    train_ds = Seq2SeqAlphaDataset(data_dir, norm, train_folders, input_len, output_len, stride,
                                   calc_norm=True, teacher_forcing_ratio=0.5)

    val_ds = Seq2SeqAlphaDataset(data_dir, norm, val_folders, input_len, output_len, stride,
                                 calc_norm=False, external_norm=norm, teacher_forcing_ratio=0.0)
    test_ds = Seq2SeqAlphaDataset(data_dir, norm, test_folders, input_len, output_len, stride,
                                  calc_norm=False, external_norm=norm, teacher_forcing_ratio=0.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"📈 Data loading completed:")
    print(f"  Training samples: {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    print(f"  Test samples: {len(test_ds)}")


    model = ImprovedTransformerSeq2Seq(
        input_dim=5, d_model=128, nhead=8,
        num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=512, dropout=0.1,
        input_len=input_len, output_len=output_len
    ).to(device)

    print(f"🤖 Improved Transformer model with Teacher Forcing support")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")


    constraint_config = {
        'constraint_mode': 'both',
        'λ_max': 0.5,
        'warm_epochs_ratio': 0.4,
        'teacher_forcing_start': 0.8,
        'teacher_forcing_end': 0.1,
        'tolerance_params': {
            'epsilon': 0.2,
            'adaptive_epsilon': True,
            'epsilon_schedule': 'cosine',
            'initial_epsilon': 0.4,
            'final_epsilon': 0.08,
            'min_epsilon': 0.02,
            'max_epsilon': 0.5,
            'loss_type': 'balanced',
            'clip_grad': True,
            'scale_factor': 0.01
        },
        'physics_config': {
            'mode': 'hybrid',
            'num_points': 4,
            'use_critical_points': True,
            'adaptive_weighting': True,
            'use_region_weights': True
        }
    }


    model, train_history, physics_evaluator = train_with_physics_evaluation(
        model=model,
        base_model=base_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        norm=norm,
        base_model_seq_len=base_model_seq_len,
        epochs=epochs,
        lr=lr,
        patience=patience,
        model_path='best_physics_constrained_model_tf.pt',
        constraint_mode=constraint_config['constraint_mode'],
        λ_max=constraint_config['λ_max'],
        warm_epochs_ratio=constraint_config['warm_epochs_ratio'],
        tolerance_params=constraint_config['tolerance_params'],
        physics_config=constraint_config['physics_config'],
        save_dir="physics_constrained_training_tf",
        teacher_forcing_start=constraint_config['teacher_forcing_start'],
        teacher_forcing_end=constraint_config['teacher_forcing_end']
    )

    return model, train_history, test_folders, output_len, physics_evaluator


def multi_step_sequence_predict_gt_window(model, alpha_true, cond, norm, seq_len=16,
                                                 output_len=8, device='cuda'):

    model.eval()

    a_mean, a_std = norm['a_mean'], norm['a_std']
    alpha_norm = (alpha_true - a_mean) / a_std

    total_steps = len(alpha_true)
    predictions = np.zeros(total_steps)
    predictions[:seq_len] = alpha_true[:seq_len]


    prediction_counts = np.zeros(total_steps)
    prediction_counts[:seq_len] = 1

    with torch.no_grad():

        for start_idx in range(seq_len, total_steps):

            window_start = max(0, start_idx - seq_len)
            window_end = start_idx
            current_seq = alpha_norm[window_start:window_end].copy()


            if len(current_seq) < seq_len:
                padding = np.full(seq_len - len(current_seq), current_seq[0])
                current_seq = np.concatenate([padding, current_seq])


            cond_seq = np.tile(cond, (seq_len, 1))
            input_seq = np.concatenate([
                current_seq.reshape(seq_len, 1),
                cond_seq
            ], axis=1)


            input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)
            output = model(input_tensor)


            pred_norm = output[0].cpu().numpy()
            pred_denorm = pred_norm * a_std + a_mean


            next_pred = pred_denorm[0]


            predictions[start_idx] = next_pred
            prediction_counts[start_idx] = 1

    return predictions

def train_physics_constrained_model_custom_with_control(data_dir, norm, base_model, base_model_seq_len,
                                                       input_len=16, output_len=8, stride=4,
                                                       batch_size=128, epochs=30, lr=5e-4, patience=10,
                                                       train_folders=None, val_folders=None, test_folders=None,
                                                       device='cuda', seed=42,
                                                       constraint_mode='both', λ_max=0.01):

    set_seed(seed)

    print("\n" + "=" * 60)
    print("🚀 Training with custom dataset split")
    print("=" * 60)
    print(f"⚡ Constraint mode: {constraint_mode}, λ_max: {λ_max}")


    if train_folders is None or val_folders is None:
        raise ValueError("train_folders and val_folders must be provided")

    print(f"📊 Using custom dataset split:")
    print(f"  Training set: {len(train_folders)} folders")
    print(f"  Validation set: {len(val_folders)} folders")
    if test_folders:
        print(f"  Test set: {len(test_folders)} folders")


    train_ds = Seq2SeqAlphaDataset(data_dir, norm, train_folders, input_len, output_len, stride,
                                   calc_norm=True, teacher_forcing_ratio=0.5)

    val_ds = Seq2SeqAlphaDataset(data_dir, norm, val_folders, input_len, output_len, stride,
                                 calc_norm=False, external_norm=norm, teacher_forcing_ratio=0.0)


    test_ds = None
    if test_folders:
        test_ds = Seq2SeqAlphaDataset(data_dir, norm, test_folders, input_len, output_len, stride,
                                      calc_norm=False, external_norm=norm, teacher_forcing_ratio=0.0)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"📈 Data loading completed:")
    print(f"  Training samples: {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    if test_ds:
        print(f"  Test samples: {len(test_ds)}")


    model = ImprovedTransformerSeq2Seq(
        input_dim=5, d_model=128, nhead=8,
        num_encoder_layers=3, num_decoder_layers=3,
        dim_feedforward=512, dropout=0.0,
        input_len=input_len, output_len=output_len
    ).to(device)

    print(f"Improved Transformer model with Teacher Forcing support")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")


    constraint_config = {
        'constraint_mode': constraint_mode,
        'λ_max': λ_max,
        'warm_epochs_ratio': 0.4,
        'teacher_forcing_start': 0.8,
        'teacher_forcing_end': 0.1,
        'tolerance_params': {
            'epsilon': 0.2,
            'adaptive_epsilon': True,
            'epsilon_schedule': 'cosine',
            'initial_epsilon': 0.4,
            'final_epsilon': 0.08,
            'min_epsilon': 0.02,
            'max_epsilon': 0.5,
            'loss_type': 'balanced',
            'clip_grad': True,
            'scale_factor': 0.01
        },
        'physics_config': {
            'mode': 'hybrid',
            'num_points': 4,
            'use_critical_points': True,
            'adaptive_weighting': True,
            'use_region_weights': True
        }
    }


    model, train_history, physics_evaluator = train_with_physics_evaluation(
        model=model,
        base_model=base_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        norm=norm,
        base_model_seq_len=base_model_seq_len,
        epochs=epochs,
        lr=lr,
        patience=patience,
        model_path='best_physics_constrained_model_tf.pt',
        constraint_mode=constraint_config['constraint_mode'],
        λ_max=constraint_config['λ_max'],
        warm_epochs_ratio=constraint_config['warm_epochs_ratio'],
        tolerance_params=constraint_config['tolerance_params'],
        physics_config=constraint_config['physics_config'],
        save_dir="physics_constrained_training_tf",
        teacher_forcing_start=constraint_config['teacher_forcing_start'],
        teacher_forcing_end=constraint_config['teacher_forcing_end']
    )

    return model, train_history, test_folders, output_len, physics_evaluator
