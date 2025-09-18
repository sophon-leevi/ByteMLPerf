# Copyright 2023 Graphcore Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import copy
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Any, Dict
import tpu_mlir
import shutil
import numpy as np
from general_perf.backends import compile_backend
from general_perf.tools import saved_to_onnx

log = logging.getLogger("CompileBackendTPU")

class CompileBackendTPU(compile_backend.CompileBackend):
    def __init__(self):
        super().__init__()
        self.hardware_type = "TPU"
        self.need_reload = False
        self.need_quant = False
        self.current_dir = os.path.split(os.path.abspath(__file__))[0]
        self.model_config = None
        self.precision = "fp32"
        self.model_precision = "F32"
        self.mean = "0.0,0.0,0.0"
        self.scale = "1.0,1.0,1.0"
        self.pixel_format = "rgb"
        self.input_num = 200
        
    def version(self) -> str:
        """
        Return compile backend version details
        """
        return tpu_mlir.distribution
    
    def pre_optimize(self, configs: Dict[str, Any]):
        """Model pre-optimization interface.

        Requirements: Model pre-optimization
        cannot change the model format. Torch model export to ONNX is allowed.
        """
        
        return configs
    
    def compile(self, configs: Dict[str, Any], dataloader=None) -> Dict[str, Any]:
        if not self.model_config:
            self.model_config = configs
        
        self.model_info = configs["model_info"]
        self.model_format = self.model_info["model_format"]
        self.interact_info = configs["interact_info"]
        self.model_path = self.model_info["model_path"]
        self.input_names = self.model_info["inputs"].split(",")
        self.model_name = self.model_info["model"]
        if("model_precision" in self.interact_info.keys()):
            self.model_precision = self.interact_info["model_precision"]
            self.mean = self.interact_info["mean"] if "mean" in self.interact_info.keys() else self.mean
            self.scale = self.interact_info["scale"] if "scale" in self.interact_info.keys() else self.scale
            self.pixel_format = self.interact_info["pixel_format"] if "pixel_format" in self.interact_info.keys() else "rgb"
            self.input_num = self.interact_info["input_num"] if "input_num" in self.interact_info.keys() else self.input_num
        if self.model_format != "onnx" and self.model_format != "pt" and self.model_format == "saved_model":
            onnx_path = os.path.join(self.model_path, self.model_name + ".onnx")
            if os.path.exists(onnx_path):
                self.model_path = onnx_path
            else:
                saved_to_onnx.savedmodel_to_onnx(self.model_path, onnx_path)
                self.model_path = onnx_path
        self.precision=self.model_precision.upper()
        
        #compile bmodels with different bs:
        current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        origin_dir = os.getcwd()
        self.compile_dir_name = current_dir + '/compiled_models/'
        if os.path.exists(self.compile_dir_name):
            shutil.rmtree(self.compile_dir_name)
        os.mkdir(self.compile_dir_name)
        os.chdir(self.compile_dir_name)
        compile_bs = configs["workload"]["batch_sizes"]
        for batch_id,batch_size in enumerate(compile_bs):
            self.input_shapes = [self.model_info["input_shape"][self.input_names[i]].copy() for i in range(len(self.input_names))]
            for i in range(len(self.input_shapes)):
                if (self.input_names[i] != "text") or ("videobert" not in self.model_name):
                    self.input_shapes[i][0] *= batch_size
            self.input_shapes_str = "" 
            for i in range(len(self.input_shapes)):
                self.input_shapes_str += '[' + ','.join(str(num) for num in self.input_shapes[i]) + ']'
                if i < len(self.input_shapes) - 1:
                    self.input_shapes_str += ','
            gen_mlir_commands = f'model_transform \
                --model_name {self.model_name} \
                --model_def ../../{self.model_path} \
                --mean {self.mean} \
                --scale {self.scale} \
                --pixel_format {self.pixel_format} \
                --input_shapes [{self.input_shapes_str}] \
                --mlir {self.model_name}_{batch_size}b.mlir'
            gen_mlir_logs = f'./model_transform_{batch_size}b.log'

            with open(gen_mlir_logs, 'w') as logfile:
                subprocess.call(gen_mlir_commands, stdout=logfile, stderr=subprocess.STDOUT, shell=True)
            if self.precision == "INT8":
                self.dataset_path = current_dir+"/datasets/"+self.model_info["dataset_name"]+"/"+self.interact_info["dataset_path"]
                if not os.path.exists(self.dataset_path):
                    os.mkdir(self.dataset_path)
                    cali_num = self.input_num
                    for i in range(cali_num):
                        test_pack = dataloader.get_samples(i)
                        input_npz, _ = test_pack
                        np.savez(os.path.join(self.dataset_path, f"{i}.npz"), **input_npz)

                if batch_id == 0:
                    self.input_shapes_1b = [self.model_info["input_shape"][self.input_names[i]] for i in range(len(self.input_names))]
                    self.input_shapes_str_1b = "" 
                    for i in range(len(self.input_shapes_1b)):
                        self.input_shapes_str_1b += '[' + ','.join(str(num) for num in self.input_shapes_1b[i]) + ']'
                        if i < len(self.input_shapes_1b) - 1:
                            self.input_shapes_str_1b += ','
                    gen_mlir_commands = f'model_transform \
                        --model_name {self.model_name} \
                        --model_def ../../{self.model_path} \
                        --mean {self.mean} \
                        --scale {self.scale} \
                        --pixel_format {self.pixel_format} \
                        --input_shapes [{self.input_shapes_str_1b}] \
                        --mlir {self.model_name}_1b.mlir'
                    gen_mlir_logs = f'./model_transform_1b.log'
                    with open(gen_mlir_logs, 'w') as logfile:
                        subprocess.call(gen_mlir_commands, stdout=logfile, stderr=subprocess.STDOUT, shell=True)
                    run_calibration_commands = f'run_calibration {self.model_name}_1b.mlir \
                        --dataset {self.dataset_path} \
                        --input_num {self.input_num}  \
                        -o {self.model_name}_cali_table'
                    run_calibration_logs = './run_calibration.log'
                    with open(run_calibration_logs , 'w') as logfile:
                        subprocess.call(run_calibration_commands, stdout=logfile, stderr=subprocess.STDOUT, shell=True)
            
                deploy_commands = f'model_deploy \
                    --mlir {self.model_name}_{batch_size}b.mlir \
                    --quantize {self.model_precision} \
                    --chip bm1690 \
                    --calibration_table {self.model_name}_cali_table \
                    --model {self.model_name}_{self.model_precision.lower()}_{batch_size}b.bmodel'
            else:
                deploy_commands = f'model_deploy \
                    --mlir {self.model_name}_{batch_size}b.mlir \
                    --quantize {self.model_precision} \
                    --chip bm1690 \
                    --model {self.model_name}_{self.model_precision.lower()}_{batch_size}b.bmodel'
            deploy_commands_logs = f'./model_deploy_{batch_size}b.log'
            
            with open(deploy_commands_logs, 'w') as logfile:
                subprocess.call(deploy_commands, stdout=logfile, stderr=subprocess.STDOUT, shell=True)
        
        os.chdir(origin_dir)
        
        result = {
            "model": self.model_name,
            "framework": configs["model_info"]["framework"],
            "compile_precision": self.precision,
            "input_type": configs["model_info"]["input_type"].split(","),
            "max_batch_size": configs["model_info"]["max_batch_size"],
            "compile_status": "success",
            "optimizations": {},
            "instance_count": 1,
            "device_count": 1,
            "sg_percent": 100,
            "segments": [
                {
                    "sg_idx": 0,
                    "is_fallback": False,
                    "input_tensor_map": configs["model_info"]["input_shape"],
                    "output_tensor_map": configs["model_info"]["outputs"],
                    "compiled_model": [
                        {
                            "compiled_bs": 1,
                            "compiled_obj": configs["model_info"]["model_path"],
                        },
                    ],
                },
            ],
            "interact_info": self.interact_info,
        }
        
        return result
    
    def get_interact_profile(self, config: Dict[str, Any]):
        """Collect information for core engine to let user interactively fill in configurations."""
        # load the interact_info by model name
        model_profile = []

        interact_info_file = os.path.join(
            self.current_dir, "interact_infos", config["model_info"]["model"] + ".json"
        )
        file_path = os.path.join(self.current_dir, self.hardware_type + ".json")

        with open(interact_info_file, "r") as f:
            interact_info = json.load(f)

       

        return interact_info
    
    def get_best_batch_size(self) -> compile_backend.List[int] | None:
        return None