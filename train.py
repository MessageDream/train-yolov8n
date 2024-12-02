import yaml
import tempfile

from clearml import Task,Dataset,OutputModel


task = Task.get_task(project_name='DevOps', task_name='Yolo8n Remote Training')
output_model = OutputModel(task=task, framework="PyTorch")
params = task.get_parameters()
model_variant = params["Args/model_variant"]
tain_ds_id = params["Args/ds_id"]
epochs = int(params["Args/epochs"])


train_ds = Dataset.get(
        dataset_id=tain_ds_id
)
train_ds_dir = train_ds.get_local_copy()

data_config = {
    'path': train_ds_dir,
    'train': 'train',
    'val': 'validation',
    'test': 'test',
    'names': ['alpaca']
}

# 保存字典为临时 YAML 文件
with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
    yaml.dump(data_config, temp_file)
    temp_yaml_path = temp_file.name

args = dict(data=temp_yaml_path, epochs=epochs)

from ultralytics import YOLO
model = YOLO(f"{model_variant}.yaml")  # build a new model from scratch

results = model.train(**args)

# 上传训练完成的 PyTorch 模型
output_model.update_weights(weights_filename=f"{model_variant}.pt")
# output_model.publish()  # 可选：将 PyTorch 模型发布

# 转换模型为 ONNX 格式
onnx_name = f"{model_variant}.onnx"
onnx_path = model.export(format="onnx", dynamic=True, opset=16)  # 导出 ONNX 模型

# 上传 ONNX 模型到 ClearML
output_model_onnx = OutputModel(task=task, framework="ONNX")
output_model_onnx.update_weights(weights_filename=onnx_name)
# output_model_onnx.publish()  # 可选：将 ONNX 模型发布

# 上传 ONNX 模型作为任务的附加 Artifact
task.upload_artifact(name="onnx_model", artifact_object=onnx_path)

print("Training and model export complete. Models uploaded to ClearML.")


