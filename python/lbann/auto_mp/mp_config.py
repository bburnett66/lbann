
import lbann

# Convenient bash to get all layers from the protobuf definition
# awk 'NR >= 138 && NR <= 231' src/proto/layers.proto | awk '{print $1}' > python/lbann/auto_mp/mp_config.py

class Config:
	"""
	Allow Lists:

	These lists signal the auto_mp procedure that it is safe 
	to set the datatype of layers to the specified precision.
	A user can add/remove layers to these lists as they see
	fit. 
	Layers not included here will be ignored and left at the
	default precision.
	It is my current understanding that setting the layers
	type sets the working precision during training and setting
	the type of the weights sets the main copy of weights used
	with the optimizer (6-20-22)
	"""
	# Allow List of layers safe to be set to half precision
	fp16_allow_list = [
		# Learning Layers
		'FullyConnected',
		'Convolution',

		# Math Layers
		'MatMul',

		# Activation Layers
		'Relu',
	]
	# Allow List of layers safe to be set to single precision
	fp32_allow_list = [
		# Loss Layers
		'CrossEntropy',
		'MeanSquaredError',
		'MeanAbsoluteError',
		'CategoricalAccuracy',
		'TopKCategoricalAccuracy',
		'L2Norm2',
		'L1Norm',

		# Regularization Layers
		'BatchNormalization',
		'LocalResponseNormalization',
		'EntrywiseBatchNormalization',
		'LayerNorm',
		'InstanceNorm',

		# Activation Layers
		'Elu',
		'LogSoftmax',
		'Softmax',
	] 

	# Run the model mutation algorithm in print only mode
	_dry_run = True

	# Force using fp16 onto the GPU
	_fp16_use_gpu = False # Currently buggy

	# Data type for the main copy of weights
	# datatypes: https://github.com/LLNL/lbann/blob/develop/src/proto/datatype.proto
	_model_weights_type = lbann.DataType.FP16

	# Set whether to use tensor cores for convolution layers
	_conv_use_tensor_core = True 

	#Add/remove layers from the config lists.
	def add_fp16_layer(self, layer_name):
		if layer_name not in self.fp16_allow_list:
			self.fp16_allow_list.append(layer_name)

	def add_fp32_layer(self, layer_name):
		if layer_name not in self.fp32_allow_list:
			self.fp32_allow_list.append(layer_name)

	def rm_fp16_layer(self, layer_name):
		if layer_name in self.fp16_allow_list:
			self.fp16_allow_list.remove(layer_name)

	def rm_fp32_layer(self, layer_name):
		if layer_name in self.fp32_allow_list:
			self.fp32_allow_list.remove(layer_name)

