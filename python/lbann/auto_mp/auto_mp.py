from . import mp_config
import lbann
import lbann.models

default_config = mp_config.Config()

def mp_model(model, config=default_config):
	new_model = model

	for l, w in zip(new_model.layers, new_model.weights):
		if l.datatype:
			#If the datatype is already set by the user then skip it
			continue

		if len(l.children) == 0:
			"""
			FIXME?
			If the layer has no children then don't alter the datatype.
			From what I understand the end layers of a model should not
			be altered, but I'm not 100% confident on this.
			"""
			continue

		# 
		if l.__class__.__name__ in config.fp16_allow_list:
			if config._dry_run:
				print(f"Modifying layer {l.__class__.__name__} to be fp16")
			else:
				# Update the layers datatype, weights datatype should be different
				l.datatype = lbann.DataType.FP16 
				w.datatype = config._model_weights_type
			if config._fp16_use_gpu:
				if config._dry_run:
					print(f"Modifying same layer's device to be gpu")
				else:
					# Update the layers device
					l.device = lbann.DeviceAllocation.GPU 
			if config._dry_run:
				print() #new line for more readable dry run

		if l.__class__.__name__ in config.fp32_allow_list:
			if config._dry_run:
				print(f"Modifying layer {l.__class__.__name__} to be fp32")
			else:
				# Update the layers datatype, weights might be different
				l.datatype = lbann.DataType.FLOAT
				w.datatype = config._model_weights_type
			if config._dry_run:
				print() #new line for more readable dry run

	return new_model 
