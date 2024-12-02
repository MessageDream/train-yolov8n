import yaml
import tempfile

from clearml import Task,Dataset,OutputModel


task = Task.get_task(project_name='DevOps', task_name='Yolo8n Remote Training')
output_model = OutputModel(task=task, framework="PyTorch")
params = task.get_parameters()
model_variant = params["Args/model_variant"]
tain_ds_id = params["Args/ds_id"]
epochs = int(params["Args/epochs"]) if params["Args/epochs"] else params["Args/epochs"]


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


