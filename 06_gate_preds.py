import pandas as pd
import numpy as np
import joblib
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

def build_tx_data(simul_row, tx_type, w, *, vd, vg, vs, vb):
    prefix = 'n_' if tx_type[:4]=='nmos' else 'p_'
    return {
        'temp': simul_row['temp'],
        'vdd': simul_row['vdd'],
        'w': w,
        'vd': vd,
        'vg': vg,
        'vs': vs,
        'vb': vb,
        'vth0': simul_row[f'{prefix}vth0'],
        'toxe': simul_row[f'{prefix}toxe'],
        'ndep': simul_row[f'{prefix}ndep'],
        'u0': simul_row[f'{prefix}u0'],
        'xj': simul_row[f'{prefix}xj']
    }

def get_gate_transistors(simul_row):
    inputs = simul_row['inputs']
    gate_name = simul_row['gate']
    vdd = simul_row['vdd']
    v_out = simul_row['v_out']
    v_x = simul_row['v_x']
    v_y = simul_row['v_y']
    v_nd = simul_row['v_nd']
    v_nr = simul_row['v_nr']
    match gate_name:
        case 'inv':
            return [
                {
                    'case': f"nmos_{'on' if inputs=='1' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                             f"nmos_{'on' if inputs=='1' else 'off'}", 
                                             wn,
                                             vd = v_out,
                                             vg = int(inputs)*vdd,
                                             vs = 0,
                                             vb = 0
                                             )
                },
                {
                    'case': f"pmos_{'on' if inputs=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                             f"pmos_{'on' if inputs=='0' else 'off'}", 
                                             wp,
                                             vd = v_out,
                                             vg = int(inputs)*vdd,
                                             vs = vdd,
                                             vb = vdd
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
                                                vd = v_x,
                                                vg = int(inputs[0])*vdd,
                                                vs = 0,
                                                vb = 0
                                                )
                },
                {# Bn: inputs[1]
                    'case': f"nmos_{'on' if inputs[1]=='1' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"nmos_{'on' if inputs[1]=='1' else 'off'}", 
                                                wn*2,
                                                vd = v_out,
                                                vg = int(inputs[1])*vdd,
                                                vs = v_x,
                                                vb = 0
                                                )
                },
                {# Ap: inputs[0]
                    'case': f"pmos_{'on' if inputs[0]=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"pmos_{'on' if inputs[0]=='0' else 'off'}", 
                                                wp,
                                                vd = v_out,
                                                vg = int(inputs[0])*vdd,
                                                vs = vdd,
                                                vb = vdd
                                                )
                },
                {# Bp: inputs[1]
                    'case': f"pmos_{'on' if inputs[1]=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"pmos_{'on' if inputs[1]=='0' else 'off'}", 
                                                wp,
                                                vd = v_out,
                                                vg = int(inputs[1])*vdd,
                                                vs = vdd,
                                                vb = vdd
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
                                                vd = v_out,
                                                vg = int(inputs[0])*vdd,
                                                vs = 0,
                                                vb = 0
                                                )
                },
                {# Bn: inputs[1]
                    'case': f"nmos_{'on' if inputs[1]=='1' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"nmos_{'on' if inputs[1]=='1' else 'off'}", 
                                                wn,
                                                vd = v_out,
                                                vg = int(inputs[1])*vdd,
                                                vs = 0,
                                                vb = 0
                                                )
                },
                {# Ap: inputs[0]
                    'case': f"pmos_{'on' if inputs[0]=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"pmos_{'on' if inputs[0]=='0' else 'off'}", 
                                                wp*2,
                                                vd = v_y,
                                                vg = int(inputs[0])*vdd,
                                                vs = vdd,
                                                vb = vdd
                                                )
                },
                {# Bp: inputs[1]
                    'case': f"pmos_{'on' if inputs[1]=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"pmos_{'on' if inputs[1]=='0' else 'off'}", 
                                                wp*2,
                                                vd = v_out,
                                                vg = int(inputs[1])*vdd,
                                                vs = v_y,
                                                vb = vdd
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
                                                vd = v_x,
                                                vg = int(inputs[0])*vdd,
                                                vs = 0,
                                                vb = 0
                                                )
                },
                {# Bn: inputs[1]
                    'case': f"nmos_{'on' if inputs[1]=='1' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"nmos_{'on' if inputs[1]=='1' else 'off'}", 
                                                wn*2,
                                                vd = v_nd,
                                                vg = int(inputs[1])*vdd,
                                                vs = v_x,
                                                vb = 0
                                                )
                },
                {# Ap: inputs[0]
                    'case': f"pmos_{'on' if inputs[0]=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"pmos_{'on' if inputs[0]=='0' else 'off'}", 
                                                wp,
                                                vd = v_nd,
                                                vg = int(inputs[0])*vdd,
                                                vs = vdd,
                                                vb = vdd
                                                )
                },
                {# Bp: inputs[1]
                    'case': f"pmos_{'on' if inputs[1]=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"pmos_{'on' if inputs[1]=='0' else 'off'}", 
                                                wp,
                                                vd = v_nd,
                                                vg = int(inputs[1])*vdd,
                                                vs = vdd,
                                                vb = vdd
                                                )
                },
                {
                    'case': f"nmos_{'on' if inputs!='11' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                             f"nmos_{'on' if inputs=='1' else 'off'}", 
                                             wn,
                                             vd = v_out,
                                             vg = v_nd,
                                             vs = 0,
                                             vb = 0
                                             )
                },
                {
                    'case': f"pmos_{'on' if inputs=='11' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                             f"pmos_{'on' if inputs=='0' else 'off'}", 
                                             wp,
                                             vd = v_out,
                                             vg = v_nd,
                                             vs = vdd,
                                             vb = vdd
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
                                                vd = v_nr,
                                                vg = int(inputs[0])*vdd,
                                                vs = 0,
                                                vb = 0
                                                )
                },
                {# Bn: inputs[1]
                    'case': f"nmos_{'on' if inputs[1]=='1' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"nmos_{'on' if inputs[1]=='1' else 'off'}", 
                                                wn,
                                                vd = v_nr,
                                                vg = int(inputs[1])*vdd,
                                                vs = 0,
                                                vb = 0
                                                )
                },
                {# Ap: inputs[0]
                    'case': f"pmos_{'on' if inputs[0]=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"pmos_{'on' if inputs[0]=='0' else 'off'}", 
                                                wp*2,
                                                vd = v_y,
                                                vg = int(inputs[0])*vdd,
                                                vs = vdd,
                                                vb = vdd
                                                )
                },
                {# Bp: inputs[1]
                    'case': f"pmos_{'on' if inputs[1]=='0' else 'off'}",
                    'tx_data': build_tx_data(simul_row, 
                                                f"pmos_{'on' if inputs[1]=='0' else 'off'}", 
                                                wp*2,
                                                vd = v_nr,
                                                vg = int(inputs[1])*vdd,
                                                vs = v_y,
                                                vb = vdd
                                                )
                },
                {
                    'case': f"nmos_{'off' if inputs!='00' else 'on'}",
                    'tx_data': build_tx_data(simul_row, 
                                             f"nmos_{'off' if inputs=='00' else 'on'}", 
                                             wn,
                                             vd = v_out,
                                             vg = v_nr,
                                             vs = 0,
                                             vb = 0
                                             )
                },
                {
                    'case': f"pmos_{'off' if inputs=='00' else 'on'}",
                    'tx_data': build_tx_data(simul_row, 
                                             f"pmos_{'on' if inputs=='00' else 'off'}", 
                                             wp,
                                             vd = v_out,
                                             vg = v_nr,
                                             vs = vdd,
                                             vb = vdd
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

def predict_gate_leakage(transistors):
    static_pwr = 0
    vdd_val = transistors[0]['tx_data']['vdd']
    for tx in transistors:
        model = tx['case']
        params = tx['tx_data']
        # joblib stores model object and 
        # the object's methods can be used
        df_params = pd.DataFrame([params])
        match model:
            case 'nmos_on':
                # Id, Ig, Is, Ib
                current_preds = models['NMOS_ON'].predict(df_params)
                # (vd.id+vs.is+vg.ig+vb.ib)/vdd
            case 'nmos_off':
                current_preds = models['NMOS_OFF'].predict(floor_vds(df_params))
            case 'pmos_on':
                current_preds = models['PMOS_ON'].predict(df_params)
            case 'pmos_off':
                current_preds = models['PMOS_OFF'].predict(floor_vds(df_params))
            case _:
                pass
        static_pwr += (
            tx['tx_data']['vd']*current_preds[0][0]
            +tx['tx_data']['vg']*current_preds[0][1]
            +tx['tx_data']['vs']*current_preds[0][2]
            +tx['tx_data']['vb']*current_preds[0][3]
        )
    return static_pwr/vdd_val

def main():
    df['leak_preds'] = df.apply(lambda x: predict_gate_leakage(get_gate_transistors(x)), axis = 1)
    df['bias'] = df['leak_preds']-df['leak']
    df['error_pct'] = (np.abs(df['bias'])/df['leak'])*100

    gate_results = df.groupby(['gate', 'inputs']).agg(
        spice_median = ('leak', 'median'),
        ml_median = ('leak_preds', 'median'),
        bias_median = ('bias', 'median'),
        median_error_pct = ('error_pct', 'median'),
        p95_error_pct = ('error_pct', lambda x: np.percentile(x, 95)),
        v_out_median = ('v_out', lambda x: "Unknown" if np.isnan(m:=np.median(x)) else m),    # gate output (SPICE)
        v_x_median = ('v_x', lambda x: "Unknown" if np.isnan(m:=np.median(x)) else m),        # NAND2/AND2: between the series NMOS
        v_y_median = ('v_y', lambda x: "Unknown" if np.isnan(m:=np.median(x)) else m),        # NOR2/OR2: between the series PMOS
        v_nd_median = ('v_nd', lambda x: "Unknown" if np.isnan(m:=np.median(x)) else m),      # AND2: NAND2 output = inverter input
        v_nr_median = ('v_nr', lambda x: "Unknown" if np.isnan(m:=np.median(x)) else m)       # OR2: NOR2 output = inverter input
    ).reset_index()

    
    print(gate_results)
    gate_results.to_csv('./results/gate_results_table.csv', index = False)
    
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