from . import mp_config
import lbann
import lbann.models

default_config = mp_config.Config()

def mp_model(model, config=default_config):
	new_model = model

	for l, w in zip(new_model.layers, new_model.weights):
		if config._dry_run:
			print(f"Layer:  {l.__class__.__name__}")
		if l.datatype:
			#If the datatype is already set by the user then skip it
			if config._dry_run:
				print(f"Layer type already set by user. Skipping")
				print() #new line for more readable dry run
			continue

		if len(l.children) == 0:
			"""
			FIXME?
			If the layer has no children then don't alter the datatype.
			From what I understand the end layers of a model should not
			be altered, but I'm not 100% confident on this.
			"""
			if config._dry_run:
				print(f"Output layer not modified")
				print() #new line for more readable dry run
			continue

		# 
		if l.__class__.__name__ in config.fp16_allow_list:
			if config._dry_run:
				print(f"Modifying layer to be fp16")
				if config._conv_use_tensor_core and (l.__class__.__name__ == 'Convolution'):
					print(f"Layer will use tensor cores")
				else:
					print(f"Config: {config._conv_use_tensor_core} name: {l.__class__.__name__}")
					print(f"Layer will NOT use tensor cores")
			else:
				# Update the layers datatype, weights datatype should be different
				l.datatype = lbann.DataType.FP16 
				w.datatype = config._model_weights_type
				if config._conv_use_tensor_core and (l.__class__.__name__ == 'Convolution'):
					l.conv_tensor_op_mode.USE_TENSOR_OPS
			"""
			#FIXME
			# Python complains that lbann.DeviceAllocation.GPU is an int 
			# and not a bytes object when creating the experiment protobuf.
			# Layers/weights are just fine, so I might be using the wrong
			# protobuf for devices
			if config._fp16_use_gpu:
				if config._dry_run:
					print(f"Modifying same layer's device to be gpu")
				else:
					# Update the layers device
					l.device = lbann.DeviceAllocation.GPU 
			"""

		if l.__class__.__name__ in config.fp32_allow_list:
			if config._dry_run:
				print(f"Modifying layer to be fp32")
			else:
				# Update the layers datatype, weights might be different
				l.datatype = lbann.DataType.FLOAT
				w.datatype = lbann.DataType.FLOAT
		
		if config._dry_run:
			print() #new line for more readable dry run

	return new_model 
