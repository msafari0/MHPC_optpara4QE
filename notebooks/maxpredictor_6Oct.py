from flask import Flask, request, jsonify, render_template
import os
import sys
sys.path.append("../src")
import pandas as pd
import ann_model as ann
from qe_parser import parse_qe_input, parse_upf, flatten_for_nn
import joblib
import numpy as np
from flask_cors import CORS
import warnings
from pymatgen.io.espresso.utils import IbravUntestedWarning

warnings.filterwarnings("ignore", category=IbravUntestedWarning)

app = Flask(__name__)

#CORS(app)
#That way, random sites cannot POST files to backend.
#If frontend + backend are served together (same domain, same port) → it doesn’t strictly need CORS(app).
#If frontend is on a different origin (e.g. maxpredict.cineca.it for frontend, API on api.cineca.it) → it absolutely needs CORS enabled.
##################### 19 sep 2025 Mandana Safari
#CORS(app, resources={r"/*": {"origins": "https://maxpredict.cineca.it"}})

@app.route('/')
def home():
    return render_template('index.html')
    
supercomputer_limits = {
    "Leonardo_Booster": {"max_cores": 2048, "max_nodes": 64, "max_threads": 20},
    "Galileo100": {"max_cores": 256, "max_nodes": 16, "max_threads": 8},
    "Marconi100": {"max_cores": 512, "max_nodes": 32, "max_threads": 8}
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_name = request.args.get('model', 'Leonardo_Booster')
        model_path = f"models/model_{model_name}.keras"
        if not os.path.exists(model_path):
            return jsonify({"error": f"Model '{model_name}' not found"}), 400

        model = ann.TimePerCall.load(model_path)

        data = None
        
        # Temporary directory for uploaded files
        tmpdir = "./tmp"
        os.makedirs(tmpdir, exist_ok=True)        

        # --- Case 1: QE input file + pseudos uploaded ---
        if 'input_file' in request.files:
            input_file = request.files['input_file']
          
            pseudos = request.files.getlist('pseudos')

            # Save temp files
            #°°°°°tmpdir = "./tmp"
            input_path = os.path.join(tmpdir, input_file.filename)
            input_file.save(input_path)

            pseudo_paths = []
            for pf in pseudos:
                pseudo_path = os.path.join(tmpdir, pf.filename)
                pf.save(pseudo_path)
                pseudo_paths.append(pseudo_path)
            
            # Build pseudo_map automatically from pseudo files
            pseudo_map = {}
            for path in pseudo_paths:
                val, nproj = parse_upf(path)
                pseudo_map[os.path.basename(path)] = {
                    "valence": val,
                    "nproj": nproj,
                    "path": path
                }

            print("DEBUG pseudo_files =", pseudo_paths, type(pseudo_paths))
            print("[DEBUG] Parsed pseudo_map =", pseudo_map)
                
            #parsed = parse_qe_input(input_path, pseudo_files=pseudo_paths)

            # Parse QE input using new version
            parsed_features = parse_qe_input(
                input_path,
                pseudo_map=pseudo_map,
                pseudos_dir=tmpdir
            )
     
            print("[DEBUG] Parsed QE features =", parsed_features)

            # --- Collect user-provided compute resources ---
            n_cores = int(request.form.get("n_cores", 0))
            n_nodes = int(request.form.get("n_nodes", 0))
            threads_per_node = int(request.form.get("threads_per_node", 0))
            n_pool = int(request.form.get("n_pool", 0))

            # Validate against supercomputer limits
            limits = supercomputer_limits.get(model_name, {})
            if n_cores > limits.get("max_cores", n_cores):
                return jsonify({"error": f"n_cores exceeds maximum for {model_name}"}), 400
            if n_nodes > limits.get("max_nodes", n_nodes):
                return jsonify({"error": f"n_nodes exceeds maximum for {model_name}"}), 400
            if threads_per_node > limits.get("max_threads", threads_per_node):
                return jsonify({"error": f"threads_per_node exceeds maximum for {model_name}"}), 400

            # Flatten for NN
            data = flatten_for_nn(parsed_features, extra_inputs={
                "n_cores": n_cores,
                "n_nodes": n_nodes,
                "threads_per_node": threads_per_node,
                "n_pool": n_pool
            })

        # --- Case 2: Raw JSON input (manual form) ---
        else:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No input data provided"}), 400
                
        # --- Filter out non-feature fields ---
        excluded_fields = {"supercomputer", "csrf_token", "model", "submit"}
        clean_data = {}
        for k, v in data.items():
            if k in excluded_fields:
                continue
            try:
                clean_data[k] = float(v)
            except (TypeError, ValueError):
                clean_data[k] = 0.0                                

        # --- Debug: print the cleaned features ---
        print("\n[DEBUG] Final features for NN input:")
        for k, v in sorted(clean_data.items()):
            print(f"  {k}: {v}")
        
        ###¯¯¯¯¯¯¯¯¯¯¯¯¯¯
        # Build consistent feature order
        expected_cols = [
            "n_el", "n_el^3", "n_species", "n_at", "n_transition",
            "n_lanthanid", "n_ks", "n_g_smooth", "n_k", "n_betas",
            "n_cores", "n_nodes", "threads_per_node", "n_pool"
        ]
        # --- Prepare for prediction ---
        test_input = pd.DataFrame([clean_data])
        #numeric_input = test_input.select_dtypes(include=[np.number])
        numeric_input = test_input.reindex(columns=expected_cols, fill_value=0.0).astype(float)
        print("[DEBUG] Columns order:", list(numeric_input.columns))
        print("[DEBUG] Sample row:", numeric_input.iloc[0].to_dict())
        prediction_normed = model.predict_normed(numeric_input)

        output = float(prediction_normed[0][0])
        return jsonify({
            "parsed_features": clean_data,
            "prediction_time": round(output, 10)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    print('Welcome Message:')
    print('\tFlask code serves as prediction-serving system for QE as of the  MAX  project.')
    print('\tArguments: YAML file which includes time_unit, position of structure name in path, and complete path to each benchmark')
    print('\t Author: Mandana Safari \n')
    app.run(debug=True)
######################
