import os
import sys
import json
import glob
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 对Q-learning算法本身进行评估并生成评估图表，此处可替换'q_eval_*.json'为前序Q-learning生成的JSON文件，参考所提供的'q_eval_PileHiddenDefect*.json'

def find_eval_file(provided=None):
    if provided and os.path.isfile(provided):
        return provided
    # look for q_eval_*.json in cwd
    matches = sorted(glob.glob('q_eval_*.json'))
    if matches:
        return matches[0]
    # fallback: any *.json that looks like an eval
    for f in glob.glob('*.json'):
        if 'eval' in f or 'q_eval' in f:
            return f
    return None


def load_eval(path):
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def extract_series(eval_data):
    # TD absolute error per episode
    td_keys = ['td_abs', 'td_loss', 'td_errors', 'td_abs_per_episode', 'td_error']
    td = None
    for k in td_keys:
        if k in eval_data:
            td = eval_data[k]
            break

    # Episode rewards
    reward_keys = ['episode_rewards', 'rewards', 'episode_reward', 'episode_total_rewards']
    rewards = None
    for k in reward_keys:
        if k in eval_data:
            rewards = eval_data[k]
            break

    # If stats nested under 'stats' or 'training'
    if td is None and 'stats' in eval_data:
        stats = eval_data['stats']
        for k in td_keys:
            if k in stats:
                td = stats[k]
                break
        for k in reward_keys:
            if k in stats:
                rewards = stats[k]
                break

    # Ensure lists
    if td is None:
        td = []
    if rewards is None:
        rewards = []
    return td, rewards


def moving_average(x, w=5):
    if not x:
        return []
    w = max(1, int(w))
    ret = []
    s = 0.0
    for i in range(len(x)):
        s += x[i]
        if i >= w:
            s -= x[i-w]
            ret.append(s / w)
        else:
            ret.append(s / (i+1))
    return ret


def plot_td(td, outpath, smooth=5, logscale=False):
    plt.figure(figsize=(10,4))
    if not td:
        plt.text(0.5,0.5,'No TD data found in eval file',ha='center',va='center')
        plt.axis('off')
    else:
        x = list(range(1, len(td)+1))
        plt.plot(x, td, label='TD abs per episode', alpha=0.4)
        ma = moving_average(td, smooth)
        plt.plot(x, ma, label=f'ma({smooth})', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('TD Absolute Error')
        plt.title('TD Absolute Error per Episode')
        plt.grid(True, alpha=0.3)
        plt.legend()
        if logscale:
            plt.yscale('log')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def find_paths_file_for_eval(eval_file):
    base = os.path.splitext(os.path.basename(eval_file))[0]
    # derive target from q_eval_<target>
    if base.startswith('q_eval_'):
        target = base[len('q_eval_'):]
    else:
        target = base

    candidates = []
    candidates.append(f'q_paths_{target}.json')
    # also try any q_paths file containing the target
    for f in glob.glob('q_paths_*.json'):
        if target and target in f:
            candidates.append(f)
    # fall back to any q_paths_*.json
    if not candidates:
        candidates = glob.glob('q_paths_*.json')

    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def load_paths(path):
    with open(path, 'r', encoding='utf-8') as fh:
        data = json.load(fh)
    # paths may be a list or dict with 'paths'
    if isinstance(data, dict):
        if 'paths' in data and isinstance(data['paths'], list):
            return data['paths']
        # maybe stored as {'top_paths': [...]} or similar
        for k in ['top_paths', 'results', 'ranked_paths']:
            if k in data and isinstance(data[k], list):
                return data[k]
        # otherwise treat dict as single entry
        return [data]
    elif isinstance(data, list):
        return data
    else:
        return []


def prepare_top_paths(paths, top_k=10):
    # Each path entry may have 'q_sum' or 'score' or 'q'
    scored = []
    for p in paths:
        if isinstance(p, dict):
            qsum = None
            for k in ['q_sum', 'qscore', 'score', 'q_total', 'q']:
                if k in p:
                    qsum = p[k]
                    break
            # if path has nested 'meta' or similar
            if qsum is None and 'meta' in p and isinstance(p['meta'], dict):
                for k in ['q_sum','score']:
                    if k in p['meta']:
                        qsum = p['meta'][k]
                        break
            # finally fallback to length as tiebreaker
            if qsum is None:
                qsum = 0.0
            scored.append((float(qsum), p))
        else:
            # unknown format
            scored.append((0.0, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def plot_delta_q(top_scored, outpath, top_k=10):
    # top_scored: list of (qsum, path_obj)
    # try to extract per-step q sequences
    has_seq = False
    seqs = []
    labels = []
    for idx, (qsum, p) in enumerate(top_scored):
        qseq = None
        if isinstance(p, dict):
            for k in ['q_values','q_per_step','q_sequence','q_seq','q_step_values','q_vals']:
                if k in p:
                    qseq = p[k]
                    break
            # sometimes stored under 'meta' or 'extra'
            if qseq is None:
                for parent in ['meta','extra','info']:
                    if parent in p and isinstance(p[parent], dict):
                        for k in ['q_values','q_per_step','q_sequence']:
                            if k in p[parent]:
                                qseq = p[parent][k]
                                break
        if qseq and isinstance(qseq, list) and len(qseq) >= 2:
            has_seq = True
            # compute delta between successive q values
            deltas = [qseq[i+1]-qseq[i] for i in range(len(qseq)-1)]
            seqs.append(deltas)
            # label from path nodes if available
            label = None
            if isinstance(p, dict):
                if 'path' in p and isinstance(p['path'], list):
                    label = '->'.join([str(x) for x in p['path'][:6]])
            if not label:
                label = f'path#{idx+1}'
            labels.append(label)

    plt.figure(figsize=(10,5))
    if has_seq:
        # plot each delta sequence
        for i, deltas in enumerate(seqs):
            x = list(range(1, len(deltas)+1))
            plt.plot(x, deltas, marker='o', label=labels[i])
        plt.xlabel('Step index')
        plt.ylabel('ΔQ (q[t+1]-q[t])')
        plt.title('ΔQ along top-ranked paths')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize='small')
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
        return True


def plot_q_dashboard(eval_file, td, rewards, top_scored, outpath, smooth=5):
    # Create a 2x2 dashboard: TD, Rewards, ΔQ (per-step or q_sums), Top Q-sums
    fig, axes = plt.subplots(2,2, figsize=(14,9))
    ax1 = axes[0,0]
    ax2 = axes[0,1]
    ax3 = axes[1,0]
    ax4 = axes[1,1]

    # TD on ax1
    if td:
        x = list(range(1, len(td)+1))
        ax1.plot(x, td, alpha=0.4, label='TD abs')
        ma = moving_average(td, smooth)
        ax1.plot(x, ma, linewidth=2, label=f'ma({smooth})')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('TD Absolute Error')
        ax1.set_title('TD Absolute Error per Episode')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    else:
        ax1.text(0.5,0.5,'No TD data',ha='center',va='center')
        ax1.axis('off')

    # Rewards on ax2
    if rewards:
        x = list(range(1, len(rewards)+1))
        ax2.bar(x, rewards, alpha=0.6)
        ma = moving_average(rewards, smooth)
        ax2.plot(x, ma, color='k', linewidth=2, label=f'ma({smooth})')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Reward')
        ax2.set_title('Episode Rewards')
        ax2.grid(True, axis='y', alpha=0.3)
        ax2.legend()
    else:
        ax2.text(0.5,0.5,'No rewards data',ha='center',va='center')
        ax2.axis('off')

    # Left panel: Weight_sum comparison (baseline vs reward-updated) on ax3
    # collect baseline and evidence weight_sum values
    qs = [q for q,p in top_scored]
    evidence_paths = None
    target = None
    try:
        base = os.path.splitext(os.path.basename(eval_file))[0]
        if base.startswith('q_eval_'):
            target = base[len('q_eval_'):]
        else:
            target = base
    except Exception:
        target = None

    if target:
        candidates = []
        candidates.extend(sorted(glob.glob(f'q_paths_{target}*evidence*.json')))
        candidates.extend(sorted(glob.glob(f'q_paths_{target}*_evidence*.json')))
        candidates.extend(sorted(glob.glob(f'q_paths_{target}*updated*.json')))
        candidates.extend(sorted(glob.glob(f'q_paths_{target}*reward*.json')))
        for f in glob.glob('q_paths_*.json'):
            if target in f and 'evidence' in f and f not in candidates:
                candidates.append(f)
        if candidates:
            for c in candidates:
                if os.path.isfile(c):
                    evidence_paths = c
                    break

    other_scored = []
    qs_other = []
    if evidence_paths:
        try:
            other_paths = load_paths(evidence_paths)
            other_scored = prepare_top_paths(other_paths, top_k=len(qs))
            qs_other = [q for q,p in other_scored]
        except Exception:
            qs_other = []

    # Extract weight_sums for baseline and other
    def extract_weight_list(scored_list):
        out = []
        for q,p in scored_list:
            ws = None
            if isinstance(p, dict):
                for k in ['weight_sum','weights_total','weightTotal','weightSum']:
                    if k in p:
                        try:
                            ws = float(p[k])
                            break
                        except Exception:
                            pass
                if ws is None and 'meta' in p and isinstance(p['meta'], dict):
                    for k in ['weight_sum','weightTotal']:
                        if k in p['meta']:
                            try:
                                ws = float(p['meta'][k])
                                break
                            except Exception:
                                pass
            out.append(ws)
        return out

    baseline_weights = extract_weight_list(top_scored)
    other_weights = extract_weight_list(other_scored) if other_scored else []

    use_weights = any([w is not None for w in baseline_weights + other_weights])
    xw = list(range(1, len(qs)+1))
    if other_scored and len(other_scored) >= 1 and use_weights:
        left_vals = [baseline_weights[i] if i < len(baseline_weights) and baseline_weights[i] is not None else (qs[i] if i < len(qs) else 0.0) for i in range(len(xw))]
        right_vals = [other_weights[i] if i < len(other_weights) and other_weights[i] is not None else (qs_other[i] if i < len(qs_other) else 0.0) for i in range(len(xw))]
        width = 0.35
        ax3.bar([xi - width/2 for xi in xw], left_vals, width=width, label='baseline', color='tab:green', alpha=0.8)
        ax3.bar([xi + width/2 for xi in xw], right_vals, width=width, label='reward-updated', color='tab:blue', alpha=0.8)
        ax3.set_xticks(xw)
        ax3.set_xticklabels([str(i) for i in xw], rotation=0, fontsize=8)
        ax3.set_xlabel('Top paths (rank)')
        ax3.set_ylabel('Weight sum')
        ax3.set_title('Baseline vs Reward-updated Top path Weight-sums')
        ax3.legend(fontsize='small')
    else:
        # show baseline weight_sums or fall back to q_sums bars
        baseline_vals = [baseline_weights[i] if i < len(baseline_weights) and baseline_weights[i] is not None else qs[i] for i in range(len(xw))]
        ax3.bar(xw, baseline_vals, color='tab:green', alpha=0.8)
        ax3.set_xticks(xw)
        ax3.set_xticklabels([str(i) for i in xw], rotation=0, fontsize=8)
        ax3.set_xlabel('Top paths (rank)')
        ax3.set_ylabel('Weight sum' if use_weights else 'Q sum')
        ax3.set_title('Top path Weight-sums' if use_weights else 'Top path Q-sums')

    # Right panel: ΔQ (use same logic as plot_delta_q)
    # try to collect per-step sequences
    has_seq = False
    seqs = []
    labels = []
    for idx, (qsum, p) in enumerate(top_scored):
        qseq = None
        if isinstance(p, dict):
            for k in ['q_values','q_per_step','q_sequence','q_seq','q_step_values','q_vals']:
                if k in p:
                    qseq = p[k]
                    break
            if qseq is None:
                for parent in ['meta','extra','info']:
                    if parent in p and isinstance(p[parent], dict):
                        for k in ['q_values','q_per_step','q_sequence']:
                            if k in p[parent]:
                                qseq = p[parent][k]
                                break
        if qseq and isinstance(qseq, list) and len(qseq) >= 2:
            has_seq = True
            deltas = [qseq[i+1]-qseq[i] for i in range(len(qseq)-1)]
            seqs.append(deltas)
            label = None
            if isinstance(p, dict) and 'path' in p and isinstance(p['path'], list):
                label = '->'.join([str(x) for x in p['path'][:6]])
            if not label:
                label = f'path#{idx+1}'
            labels.append(label)

    if has_seq:
        for i, deltas in enumerate(seqs):
            x = list(range(1, len(deltas)+1))
            ax4.plot(x, deltas, marker='o', label=labels[i])
        ax4.set_xlabel('Step index')
        ax4.set_ylabel('ΔQ (q[t+1]-q[t])')
        ax4.set_title('ΔQ along top-ranked paths')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize='small')
    else:
        # fallback: show delta between adjacent q_sum values
        q_sums = [q for q,p in top_scored]
        vals = q_sums
        xq = list(range(1, len(vals)+1))
        deltas = [(vals[i] - vals[i+1]) if (i+1) < len(vals) else 0.0 for i in range(len(vals))]
        colors = ['tab:green' if d >= 0 else 'tab:orange' for d in deltas]
        ax4.bar(xq, deltas, color=colors, alpha=0.8)
        ax4.set_xlabel('Top paths (rank)')
        ax4.set_ylabel('ΔQ (q[i]-q[i+1])')
        ax4.set_title('ΔQ between adjacent top path Q-sums')
        ax4.set_xticks(xq)
        ax4.set_xticklabels([str(i) for i in xq], rotation=0)
        ax4.grid(axis='y', alpha=0.3)
        for i, v in enumerate(deltas):
            ax4.text(xq[i], v, f'{v:.2f}', ha='center', va='bottom' if v>=0 else 'top', fontsize=8)

    # adjust bottom margin and ensure saved image includes labels
    try:
        fig.tight_layout()
    except Exception:
        pass
    try:
        fig.subplots_adjust(bottom=0.28)
    except Exception:
        pass

    fig.suptitle(f'Q-value Dashboard: {os.path.basename(eval_file)}', fontsize=14)
    # ensure labels are included in saved image
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(outpath, bbox_inches='tight', dpi=150)
    plt.close()
    return outpath


def plot_rewards(rewards, outpath, smooth=5, logscale=False):
    plt.figure(figsize=(10,4))
    if not rewards:
        plt.text(0.5,0.5,'No episode rewards found in eval file',ha='center',va='center')
        plt.axis('off')
    else:
        x = list(range(1, len(rewards)+1))
        plt.bar(x, rewards, alpha=0.6, label='Episode reward')
        ma = moving_average(rewards, smooth)
        plt.plot(x, ma, color='k', label=f'ma({smooth})', linewidth=2)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Episode Rewards')
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend()
        if logscale:
            # allow only positive values for log
            ymin = min([r for r in rewards if r>0], default=None)
            if ymin is not None:
                plt.yscale('log')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    eval_file = find_eval_file(arg)
    if not eval_file:
        print('No eval JSON file found (looked for q_eval_*.json).')
        sys.exit(1)
    print('Evaluation loaded from', eval_file)
    data = load_eval(eval_file)
    td, rewards = extract_series(data)

    out_dir = os.path.join('.', 'eval_figures')
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(eval_file))[0]
    td_path = os.path.join(out_dir, f'{base}_td.png')
    rewards_path = os.path.join(out_dir, f'{base}_rewards.png')

    plot_td(td, td_path)
    plot_rewards(rewards, rewards_path)

    print('Saved figures:')
    print(' ', td_path)
    print(' ', rewards_path)

    # attempt to find matching q_paths file and plot delta-Q for top-ranked paths
    paths_file = find_paths_file_for_eval(eval_file)
    if paths_file:
        print('Found paths file:', paths_file)
        try:
            paths = load_paths(paths_file)
            top_scored = prepare_top_paths(paths, top_k=10)
            delta_out = os.path.join(out_dir, f'{base}_delta_q.png')
            ok = plot_delta_q(top_scored, delta_out, top_k=10)
            if ok:
                print(' ', delta_out)
            # also create a combined dashboard containing TD, rewards, ΔQ and top Q-sums
            try:
                dashboard_out = os.path.join(out_dir, f'{base}_q_dashboard.png')
                dout = plot_q_dashboard(eval_file, td, rewards, top_scored, dashboard_out)
                if dout:
                    print(' ', dashboard_out)
            except Exception:
                pass
        except Exception as e:
            print('Could not plot delta Q:', e)
    else:
        print('No q_paths_*.json found for this evaluation; skipping delta-Q plot.')


if __name__ == '__main__':
    main()
