from . import mp_config
import lbann
import lbann.models

default_config = mp_config.Config()

def mp_model(model, config=default_config):
	layers = model.layers

	for l in layers:
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
		if l.__class__.__name in config.fp16_allow_list:
			if config._dry_run:
				print(f"Modifying layer {l.__class__.__name__} to be fp16")
			else:
				#TODO: Update the layers datatype
				print("TODO")
			if config._fp16_use_gpu:
				if config._dry_run:
					print(f"Modifying same layer's device to be gpu")
				else:
					#TODO: Update the layers device
					print("TODO")

		if l.__class__.__name in config.fp32_allow_list:
			if config._dry_run:
				print(f"Modifying layer {l.__class__.__name__} to be fp32")
			else:
				#TODO: Update the layers datatype
				print("TODO")
		
	    print(f'layer:\t\t{l.__class__.__name__}')
	    print(f'Data type:\t{l.datatype}')
	    print(f'Device:\t\t{l.device}')
	    print(f'Num children:\t{len(l.children)}')
	    print()

	model.layers = layers

	#TODO Modify the objective function. Not quite sure how to do this yet...
