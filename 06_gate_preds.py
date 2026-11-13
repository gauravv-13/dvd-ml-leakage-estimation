"""Phase 5 (Part B steps 4-5): ML leakage of the five gates, 18 input cases x 200 corners, vs SPICE.

For every row of data/gates/gate_spice.csv, every transistor of the gate is run through the chosen
(MLP) model of its case, with terminal voltages from the SPICE node voltages of that row (spec step 4).
Each transistor entry names the node of each terminal ('out', 'x', 'y', 'nd', 'nr', the inputs 'a'/'b',
'vdd' or '0'); input naming as in the course slides: A drives the transistor next to the rail.

Leakage (same definition as the SPICE reference, static power / VDD = current from the VDD supply +
current from the inputs held at VDD):
  1. ON transistors: Ig and Ib from the ON model; Id and Is from Kirchhoff's current law at the
     internal and output nodes (an ON transistor passes on what the OFF transistors leak; its own
     channel current would depend on the few-uV drop across it). Nodes with the fewest unknown ON
     channels are solved first; parallel ON transistors share a node current by width.
  2. Leakage = sum of the currents into the terminals on 'vdd' and on inputs held at 1.
Comparison columns: the same terminal sum with the ON models' own Id/Is (leak_terminal), the power sum
sum(V*I)/VDD (leak_vi), and all three with the output-type nodes (out, nd, nr) set to exactly 0 or VDD
instead of the SPICE values (*_ideal), to show how much each method relies on the SPICE voltages.
OFF transistors with |Vds| < 1 mV are evaluated at 1 mV (training data barely covers that region).

Outputs
  data/gates/gate_ml.csv            per row: SPICE node voltages and leakage, ML leakage (all variants), error
  results/gate_results_table.csv    deliverable 6 per gate and inputs (main method + comparison columns)
  results/gate_parity_plot.png      deliverable 7
Run:  python 06_gate_preds.py   (after 05_gate_spice.py)
"""
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

models = {
    'NMOS_ON': joblib.load('saved_models/mlp/nmos_on.joblib'),
    'NMOS_OFF': joblib.load('saved_models/mlp/nmos_off.joblib'),
    'PMOS_ON': joblib.load('saved_models/mlp/pmos_on.joblib'),
    'PMOS_OFF': joblib.load('saved_models/mlp/pmos_off.joblib')
}

df_corners = pd.read_csv('./data/gates/corners.csv')
df_spice = pd.read_csv('./data/gates/gate_spice.csv', dtype={"inputs": str})
df = pd.merge(df_spice, df_corners, on=['corner', 'vdd', 'temp'], how='inner')
wn=32*1e-9
wp=64*1e-9

TERMINALS = ['d', 'g', 's', 'b']          # predict() columns: Id, Ig, Is, Ib
FREE_NODES = ['out', 'x', 'y', 'nd', 'nr']   # nodes not tied to a source


def node_voltage(simul_row, node):
    """Voltage of a gate node: the SPICE node voltage, an input (exactly 0 or VDD), VDD or ground."""
    if node == 'vdd':
        return simul_row['vdd']
    if node == '0':
        return 0.0
    if node in ('a', 'b'):
        return int(simul_row['inputs']['ab'.index(node)]) * simul_row['vdd']
    return simul_row[f'v_{node}']


def build_tx_data(simul_row, tx_type, w, *, vd, vg, vs, vb):
    prefix = 'n_' if tx_type[:4]=='nmos' else 'p_'
    return {
        'temp': simul_row['temp'],
        'vdd': simul_row['vdd'],
        'w': w,
        'vd': node_voltage(simul_row, vd),
        'vg': node_voltage(simul_row, vg),
        'vs': node_voltage(simul_row, vs),
        'vb': node_voltage(simul_row, vb),
        'vth0': simul_row[f'{prefix}vth0'],
        'toxe': simul_row[f'{prefix}toxe'],
        'ndep': simul_row[f'{prefix}ndep'],
        'u0': simul_row[f'{prefix}u0'],
        'xj': simul_row[f'{prefix}xj'],
        'nodes': {'d': vd, 'g': vg, 's': vs, 'b': vb}
    }

def get_gate_transistors(simul_row, ideal=False):
    if ideal:   # output-type nodes at exactly 0 or VDD instead of the SPICE values
        simul_row = simul_row.copy()
        for n in ['v_out', 'v_nd', 'v_nr']:
            if not pd.isna(simul_row[n]):
                simul_row[n] = simul_row['vdd'] if simul_row[n] > simul_row['vdd'] / 2 else 0.0

    inputs = simul_row['inputs']
    gate_name = simul_row['gate']
    match gate_name:
        case 'inv':
            return [
                {
                    'case': f"nmos_{'on' if inputs=='1' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                             f"nmos_{'on' if inputs=='1' else 'off'}", 
                                             wn,
                                             vd = 'out',
                                             vg = 'a',
                                             vs = '0',
                                             vb = '0'
                                             )
                },
                {
                    'case': f"pmos_{'on' if inputs=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                             f"pmos_{'on' if inputs=='0' else 'off'}", 
                                             wp,
                                             vd = 'out',
                                             vg = 'a',
                                             vs = 'vdd',
                                             vb = 'vdd'
                                             )
                }
            ]
        # x-> nand/and
        # y-> nor/or
        case 'nand2':
            return [
                {# An: inputs[0]
                    'case': f"nmos_{'on' if inputs[0]=='1' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"nmos_{'on' if inputs[0]=='1' else 'off'}", 
                                                wn*2,
                                                vd = 'x',
                                                vg = 'a',
                                                vs = '0',
                                                vb = '0'
                                                )
                },
                {# Bn: inputs[1]
                    'case': f"nmos_{'on' if inputs[1]=='1' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"nmos_{'on' if inputs[1]=='1' else 'off'}", 
                                                wn*2,
                                                vd = 'out',
                                                vg = 'b',
                                                vs = 'x',
                                                vb = '0'
                                                )
                },
                {# Ap: inputs[0]
                    'case': f"pmos_{'on' if inputs[0]=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"pmos_{'on' if inputs[0]=='0' else 'off'}", 
                                                wp,
                                                vd = 'out',
                                                vg = 'a',
                                                vs = 'vdd',
                                                vb = 'vdd'
                                                )
                },
                {# Bp: inputs[1]
                    'case': f"pmos_{'on' if inputs[1]=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"pmos_{'on' if inputs[1]=='0' else 'off'}", 
                                                wp,
                                                vd = 'out',
                                                vg = 'b',
                                                vs = 'vdd',
                                                vb = 'vdd'
                                                )
                }
            ]
        case 'nor2':
            return [
                {# An: inputs[0]
                    'case': f"nmos_{'on' if inputs[0]=='1' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"nmos_{'on' if inputs[0]=='1' else 'off'}", 
                                                wn,
                                                vd = 'out',
                                                vg = 'a',
                                                vs = '0',
                                                vb = '0'
                                                )
                },
                {# Bn: inputs[1]
                    'case': f"nmos_{'on' if inputs[1]=='1' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"nmos_{'on' if inputs[1]=='1' else 'off'}", 
                                                wn,
                                                vd = 'out',
                                                vg = 'b',
                                                vs = '0',
                                                vb = '0'
                                                )
                },
                {# Ap: inputs[0]
                    'case': f"pmos_{'on' if inputs[0]=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"pmos_{'on' if inputs[0]=='0' else 'off'}", 
                                                wp*2,
                                                vd = 'y',
                                                vg = 'a',
                                                vs = 'vdd',
                                                vb = 'vdd'
                                                )
                },
                {# Bp: inputs[1]
                    'case': f"pmos_{'on' if inputs[1]=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"pmos_{'on' if inputs[1]=='0' else 'off'}", 
                                                wp*2,
                                                vd = 'out',
                                                vg = 'b',
                                                vs = 'y',
                                                vb = 'vdd'
                                                )
                }
            ]
        # v_nd: output of intermediate nand in AND
        # v_nr: output of intermediate nor in OR
        case 'and2':
            return [
                {# An: inputs[0]
                    'case': f"nmos_{'on' if inputs[0]=='1' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"nmos_{'on' if inputs[0]=='1' else 'off'}", 
                                                wn*2,
                                                vd = 'x',
                                                vg = 'a',
                                                vs = '0',
                                                vb = '0'
                                                )
                },
                {# Bn: inputs[1]
                    'case': f"nmos_{'on' if inputs[1]=='1' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"nmos_{'on' if inputs[1]=='1' else 'off'}", 
                                                wn*2,
                                                vd = 'nd',
                                                vg = 'b',
                                                vs = 'x',
                                                vb = '0'
                                                )
                },
                {# Ap: inputs[0]
                    'case': f"pmos_{'on' if inputs[0]=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"pmos_{'on' if inputs[0]=='0' else 'off'}", 
                                                wp,
                                                vd = 'nd',
                                                vg = 'a',
                                                vs = 'vdd',
                                                vb = 'vdd'
                                                )
                },
                {# Bp: inputs[1]
                    'case': f"pmos_{'on' if inputs[1]=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"pmos_{'on' if inputs[1]=='0' else 'off'}", 
                                                wp,
                                                vd = 'nd',
                                                vg = 'b',
                                                vs = 'vdd',
                                                vb = 'vdd'
                                                )
                },
                {
                    'case': f"nmos_{'on' if inputs!='11' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                             f"nmos_{'on' if inputs=='1' else 'off'}", 
                                             wn,
                                             vd = 'out',
                                             vg = 'nd',
                                             vs = '0',
                                             vb = '0'
                                             )
                },
                {
                    'case': f"pmos_{'on' if inputs=='11' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                             f"pmos_{'on' if inputs=='0' else 'off'}", 
                                             wp,
                                             vd = 'out',
                                             vg = 'nd',
                                             vs = 'vdd',
                                             vb = 'vdd'
                                             )
                }
            ]
        case 'or2':
            return [
                {# An: inputs[0]
                    'case': f"nmos_{'on' if inputs[0]=='1' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"nmos_{'on' if inputs[0]=='1' else 'off'}", 
                                                wn,
                                                vd = 'nr',
                                                vg = 'a',
                                                vs = '0',
                                                vb = '0'
                                                )
                },
                {# Bn: inputs[1]
                    'case': f"nmos_{'on' if inputs[1]=='1' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"nmos_{'on' if inputs[1]=='1' else 'off'}", 
                                                wn,
                                                vd = 'nr',
                                                vg = 'b',
                                                vs = '0',
                                                vb = '0'
                                                )
                },
                {# Ap: inputs[0]
                    'case': f"pmos_{'on' if inputs[0]=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"pmos_{'on' if inputs[0]=='0' else 'off'}", 
                                                wp*2,
                                                vd = 'y',
                                                vg = 'a',
                                                vs = 'vdd',
                                                vb = 'vdd'
                                                )
                },
                {# Bp: inputs[1]
                    'case': f"pmos_{'on' if inputs[1]=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"pmos_{'on' if inputs[1]=='0' else 'off'}", 
                                                wp*2,
                                                vd = 'nr',
                                                vg = 'b',
                                                vs = 'y',
                                                vb = 'vdd'
                                                )
                },
                {
                    'case': f"nmos_{'off' if inputs!='00' else 'on'}",
                    'tx_data': build_tx_data(simul_row, 
                                             f"nmos_{'off' if inputs=='00' else 'on'}", 
                                             wn,
                                             vd = 'out',
                                             vg = 'nr',
                                             vs = '0',
                                             vb = '0'
                                             )
                },
                {
                    'case': f"pmos_{'off' if inputs=='00' else 'on'}",
                    'tx_data': build_tx_data(simul_row, 
                                             f"pmos_{'on' if inputs=='00' else 'off'}", 
                                             wp,
                                             vd = 'out',
                                             vg = 'nr',
                                             vs = 'vdd',
                                             vb = 'vdd'
                                             )
                }
            ]
        case _:
            pass



VDS_FLOOR = 1e-3  # V; the training data barely covers |Vds| below this

def floor_vds(df_params):
    """OFF transistors with |Vds| < 1 mV are evaluated at 1 mV (training-data coverage).
    Vd is moved 1 mV away from Vs, towards the inside of [0, VDD]; the model is unchanged."""
    df_params = df_params.copy()
    small = (df_params['vd'] - df_params['vs']).abs() < VDS_FLOOR
    step = np.where(df_params['vs'] > df_params['vdd'] / 2, -VDS_FLOOR, VDS_FLOOR)
    df_params.loc[small, 'vd'] = df_params.loc[small, 'vs'] + step[small]
    return df_params

def predict_currents(transistors):
    """Predicted currents into d, g, s, b of every transistor (dict per transistor)."""
    for tx in transistors:
        params = {k: v for k, v in tx['tx_data'].items() if k != 'nodes'}
        df_params = pd.DataFrame([params])
        match tx['case']:
            case 'nmos_on':
                current_preds = models['NMOS_ON'].predict(df_params)
            case 'nmos_off':
                current_preds = models['NMOS_OFF'].predict(floor_vds(df_params))
            case 'pmos_on':
                current_preds = models['PMOS_ON'].predict(df_params)
            case 'pmos_off':
                current_preds = models['PMOS_OFF'].predict(floor_vds(df_params))
        tx['I'] = dict(zip(TERMINALS, current_preds[0]))
    return transistors

def kcl_currents(transistors):
    """Id/Is of ON transistors from KCL at the free nodes (copy; OFF transistors and all Ig, Ib unchanged)."""
    txs = [{**tx, 'I': dict(tx['I'])} for tx in transistors]
    unknown = {(i, k) for i, tx in enumerate(txs) if tx['case'].endswith('on') for k in ('d', 's')}
    while unknown:
        options = []
        for node in FREE_NODES:
            u = [(i, k) for (i, k) in unknown if txs[i]['tx_data']['nodes'][k] == node]
            if u:
                options.append((len(u), node, u))
        _, node, u = min(options)
        known = sum(tx['I'][k] for i, tx in enumerate(txs) for k in TERMINALS
                    if tx['tx_data']['nodes'][k] == node and (i, k) not in unknown)
        w_total = sum(txs[i]['tx_data']['w'] for i, _ in u)
        for i, k in u:   # parallel ON transistors share the node current by width
            tx = txs[i]
            other = 's' if k == 'd' else 'd'
            tx['I'][k] = -known * tx['tx_data']['w'] / w_total
            tx['I'][other] = -(tx['I'][k] + tx['I']['g'] + tx['I']['b'])
            unknown -= {(i, k), (i, other)}
    return txs

def terminal_sum(transistors, inputs):
    """Current from the VDD supply + inputs at VDD = currents into the terminals on those nodes."""
    at_vdd = {'vdd'} | {n for n, bit in zip('ab', inputs) if bit == '1'}
    return sum(tx['I'][k] for tx in transistors for k in TERMINALS if tx['tx_data']['nodes'][k] in at_vdd)

def power_sum(transistors):
    """sum(V*I) over all terminals / VDD (equal to the terminal sum only if the currents obey KCL)."""
    vdd_val = transistors[0]['tx_data']['vdd']
    return sum(tx['tx_data'][f'v{k}'] * tx['I'][k] for tx in transistors for k in TERMINALS) / vdd_val

def predict_gate_leakage(simul_row):
    out = {}
    for suffix, ideal in [('', False), ('_ideal', True)]:
        transistors = predict_currents(get_gate_transistors(simul_row, ideal))
        out['leak_preds' + suffix] = terminal_sum(kcl_currents(transistors), simul_row['inputs'])
        out['leak_terminal' + suffix] = terminal_sum(transistors, simul_row['inputs'])
        out['leak_vi' + suffix] = power_sum(transistors)
    return pd.Series(out)

def err_pct(pred, spice):
    return (pred - spice) / spice * 100

def main():
    preds = df.apply(predict_gate_leakage, axis=1)
    df[preds.columns] = preds
    df['bias'] = df['leak_preds']-df['leak']
    df['error_pct'] = (np.abs(df['bias'])/df['leak'])*100
    df[['gate', 'inputs', 'corner', 'vdd', 'temp', 'v_out', 'v_x', 'v_y', 'v_nd', 'v_nr', 'leak',
        *preds.columns, 'error_pct']].to_csv('./data/gates/gate_ml.csv', index=False, float_format='%.12e')

    def median_or_unknown(x):
        return "Unknown" if np.isnan(m := np.median(x)) else m

    gate_results = df.groupby(['gate', 'inputs'], sort=False).agg(
        spice_median = ('leak', 'median'),
        ml_median = ('leak_preds', 'median'),
        bias_median = ('bias', 'median'),
        median_error_pct = ('error_pct', 'median'),
        p95_error_pct = ('error_pct', lambda x: np.percentile(x, 95)),
        v_out_median = ('v_out', median_or_unknown),    # gate output (SPICE)
        v_x_median = ('v_x', median_or_unknown),        # NAND2/AND2: between the series NMOS
        v_y_median = ('v_y', median_or_unknown),        # NOR2/OR2: between the series PMOS
        v_nd_median = ('v_nd', median_or_unknown),      # AND2: NAND2 output = inverter input
        v_nr_median = ('v_nr', median_or_unknown)       # OR2: NOR2 output = inverter input
    ).reset_index()
    groups = df.groupby(['gate', 'inputs'], sort=False)
    gate_results.insert(5, 'bias_pct', groups.apply(lambda g: err_pct(g.leak_preds, g.leak).median(), include_groups=False).to_numpy())
    gate_results.insert(8, 'mean_accuracy_pct', groups.apply(lambda g: 100 - g.error_pct.mean(), include_groups=False).to_numpy())
    for col in ['leak_terminal', 'leak_vi', 'leak_preds_ideal', 'leak_terminal_ideal', 'leak_vi_ideal']:
        name = col.replace('leak_', '')
        gate_results[f'{name}_median_error_pct'] = groups.apply(lambda g: err_pct(g[col], g.leak).abs().median(), include_groups=False).to_numpy()
        gate_results[f'{name}_mean_accuracy_pct'] = groups.apply(lambda g: 100 - err_pct(g[col], g.leak).abs().mean(), include_groups=False).to_numpy()

    print(gate_results.iloc[:, :9].to_string(index=False))
    gate_results.to_csv('./results/gate_results_table.csv', index = False)
    for col in ['leak_preds', 'leak_terminal', 'leak_vi', 'leak_preds_ideal', 'leak_terminal_ideal', 'leak_vi_ideal']:
        e = err_pct(df[col], df['leak']).abs()
        print(f"{col:20s} all rows: median {e.median():6.2f} %  p95 {e.quantile(0.95):8.2f} %  mean accuracy {100 - e.mean():9.2f} %")

    plt.figure(figsize=(7, 6))

    # Scatter plot of all data points
    plt.scatter(df['leak'], df['leak_preds'], alpha=0.3, edgecolors='none', s=20, label='Corners')

    # Reference ideal line: y = x
    min_val = min(df['leak'].min(), df['leak_preds'].min())
    max_val = max(df['leak'].max(), df['leak_preds'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal (y = x)')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('SPICE Reference Leakage (A)')
    plt.ylabel('ML Predicted Leakage (A)')
    plt.title('Gate Leakage Parity Plot (Log-Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('./results/gate_parity_plot.png', dpi=300)

if __name__=="__main__":
    main()
