
°ÿ
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Á
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.11.02v2.11.0-rc2-15-g6290819256d8ã
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0

"RMSprop/velocity/Output-Layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"RMSprop/velocity/Output-Layer/bias

6RMSprop/velocity/Output-Layer/bias/Read/ReadVariableOpReadVariableOp"RMSprop/velocity/Output-Layer/bias*
_output_shapes
:*
dtype0
¤
$RMSprop/velocity/Output-Layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*5
shared_name&$RMSprop/velocity/Output-Layer/kernel

8RMSprop/velocity/Output-Layer/kernel/Read/ReadVariableOpReadVariableOp$RMSprop/velocity/Output-Layer/kernel*
_output_shapes

:*
dtype0
 
$RMSprop/velocity/Hidden-Layer-2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$RMSprop/velocity/Hidden-Layer-2/bias

8RMSprop/velocity/Hidden-Layer-2/bias/Read/ReadVariableOpReadVariableOp$RMSprop/velocity/Hidden-Layer-2/bias*
_output_shapes
:*
dtype0
¨
&RMSprop/velocity/Hidden-Layer-2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&RMSprop/velocity/Hidden-Layer-2/kernel
¡
:RMSprop/velocity/Hidden-Layer-2/kernel/Read/ReadVariableOpReadVariableOp&RMSprop/velocity/Hidden-Layer-2/kernel*
_output_shapes

:*
dtype0
 
$RMSprop/velocity/Hidden-Layer-1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*5
shared_name&$RMSprop/velocity/Hidden-Layer-1/bias

8RMSprop/velocity/Hidden-Layer-1/bias/Read/ReadVariableOpReadVariableOp$RMSprop/velocity/Hidden-Layer-1/bias*
_output_shapes
:*
dtype0
¨
&RMSprop/velocity/Hidden-Layer-1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*7
shared_name(&RMSprop/velocity/Hidden-Layer-1/kernel
¡
:RMSprop/velocity/Hidden-Layer-1/kernel/Read/ReadVariableOpReadVariableOp&RMSprop/velocity/Hidden-Layer-1/kernel*
_output_shapes

:*
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	iterationVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	iteration
_
iteration/Read/ReadVariableOpReadVariableOp	iteration*
_output_shapes
: *
dtype0	
z
Output-Layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameOutput-Layer/bias
s
%Output-Layer/bias/Read/ReadVariableOpReadVariableOpOutput-Layer/bias*
_output_shapes
:*
dtype0

Output-Layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*$
shared_nameOutput-Layer/kernel
{
'Output-Layer/kernel/Read/ReadVariableOpReadVariableOpOutput-Layer/kernel*
_output_shapes

:*
dtype0
~
Hidden-Layer-2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameHidden-Layer-2/bias
w
'Hidden-Layer-2/bias/Read/ReadVariableOpReadVariableOpHidden-Layer-2/bias*
_output_shapes
:*
dtype0

Hidden-Layer-2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameHidden-Layer-2/kernel

)Hidden-Layer-2/kernel/Read/ReadVariableOpReadVariableOpHidden-Layer-2/kernel*
_output_shapes

:*
dtype0
~
Hidden-Layer-1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameHidden-Layer-1/bias
w
'Hidden-Layer-1/bias/Read/ReadVariableOpReadVariableOpHidden-Layer-1/bias*
_output_shapes
:*
dtype0

Hidden-Layer-1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*&
shared_nameHidden-Layer-1/kernel

)Hidden-Layer-1/kernel/Read/ReadVariableOpReadVariableOpHidden-Layer-1/kernel*
_output_shapes

:*
dtype0

$serving_default_Hidden-Layer-1_inputPlaceholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
É
StatefulPartitionedCallStatefulPartitionedCall$serving_default_Hidden-Layer-1_inputHidden-Layer-1/kernelHidden-Layer-1/biasHidden-Layer-2/kernelHidden-Layer-2/biasOutput-Layer/kernelOutput-Layer/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference_signature_wrapper_1759

NoOpNoOp
'
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Î&
valueÄ&BÁ& Bº&
Á
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¦
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias*
¦
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias*
.
0
1
2
3
#4
$5*
.
0
1
2
3
#4
$5*
* 
°
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses*
6
*trace_0
+trace_1
,trace_2
-trace_3* 
6
.trace_0
/trace_1
0trace_2
1trace_3* 
* 

2
_variables
3_iterations
4_learning_rate
5_index_dict
6_velocities
7
_momentums
8_average_gradients
9_update_step_xla*

:serving_default* 

0
1*

0
1*
* 

;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

@trace_0* 

Atrace_0* 
e_
VARIABLE_VALUEHidden-Layer-1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEHidden-Layer-1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

Gtrace_0* 

Htrace_0* 
e_
VARIABLE_VALUEHidden-Layer-2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
a[
VARIABLE_VALUEHidden-Layer-2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

#0
$1*

#0
$1*
* 

Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*

Ntrace_0* 

Otrace_0* 
c]
VARIABLE_VALUEOutput-Layer/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
_Y
VARIABLE_VALUEOutput-Layer/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

P0
Q1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
5
30
R1
S2
T3
U4
V5
W6*
SM
VARIABLE_VALUE	iteration0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUElearning_rate3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
R0
S1
T2
U3
V4
W5*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
X	variables
Y	keras_api
	Ztotal
	[count*
H
\	variables
]	keras_api
	^total
	_count
`
_fn_kwargs*
qk
VARIABLE_VALUE&RMSprop/velocity/Hidden-Layer-1/kernel1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$RMSprop/velocity/Hidden-Layer-1/bias1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUE&RMSprop/velocity/Hidden-Layer-2/kernel1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$RMSprop/velocity/Hidden-Layer-2/bias1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE$RMSprop/velocity/Output-Layer/kernel1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE"RMSprop/velocity/Output-Layer/bias1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUE*

Z0
[1*

X	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

^0
_1*

\	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
³
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)Hidden-Layer-1/kernel/Read/ReadVariableOp'Hidden-Layer-1/bias/Read/ReadVariableOp)Hidden-Layer-2/kernel/Read/ReadVariableOp'Hidden-Layer-2/bias/Read/ReadVariableOp'Output-Layer/kernel/Read/ReadVariableOp%Output-Layer/bias/Read/ReadVariableOpiteration/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp:RMSprop/velocity/Hidden-Layer-1/kernel/Read/ReadVariableOp8RMSprop/velocity/Hidden-Layer-1/bias/Read/ReadVariableOp:RMSprop/velocity/Hidden-Layer-2/kernel/Read/ReadVariableOp8RMSprop/velocity/Hidden-Layer-2/bias/Read/ReadVariableOp8RMSprop/velocity/Output-Layer/kernel/Read/ReadVariableOp6RMSprop/velocity/Output-Layer/bias/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpConst*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *&
f!R
__inference__traced_save_1980
Æ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameHidden-Layer-1/kernelHidden-Layer-1/biasHidden-Layer-2/kernelHidden-Layer-2/biasOutput-Layer/kernelOutput-Layer/bias	iterationlearning_rate&RMSprop/velocity/Hidden-Layer-1/kernel$RMSprop/velocity/Hidden-Layer-1/bias&RMSprop/velocity/Hidden-Layer-2/kernel$RMSprop/velocity/Hidden-Layer-2/bias$RMSprop/velocity/Output-Layer/kernel"RMSprop/velocity/Output-Layer/biastotal_1count_1totalcount*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *)
f$R"
 __inference__traced_restore_2044å¯

¶
D__inference_sequential_layer_call_and_return_conditional_losses_1585

inputs%
hidden_layer_1_1545:!
hidden_layer_1_1547:%
hidden_layer_2_1562:!
hidden_layer_2_1564:#
output_layer_1579:
output_layer_1581:
identity¢&Hidden-Layer-1/StatefulPartitionedCall¢&Hidden-Layer-2/StatefulPartitionedCall¢$Output-Layer/StatefulPartitionedCall
&Hidden-Layer-1/StatefulPartitionedCallStatefulPartitionedCallinputshidden_layer_1_1545hidden_layer_1_1547*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_1544«
&Hidden-Layer-2/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-1/StatefulPartitionedCall:output:0hidden_layer_2_1562hidden_layer_2_1564*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_1561£
$Output-Layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-2/StatefulPartitionedCall:output:0output_layer_1579output_layer_1581*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Output-Layer_layer_call_and_return_conditional_losses_1578|
IdentityIdentity-Output-Layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
NoOpNoOp'^Hidden-Layer-1/StatefulPartitionedCall'^Hidden-Layer-2/StatefulPartitionedCall%^Output-Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2P
&Hidden-Layer-1/StatefulPartitionedCall&Hidden-Layer-1/StatefulPartitionedCall2P
&Hidden-Layer-2/StatefulPartitionedCall&Hidden-Layer-2/StatefulPartitionedCall2L
$Output-Layer/StatefulPartitionedCall$Output-Layer/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
å$
²
__inference__wrapped_model_1526
hidden_layer_1_inputJ
8sequential_hidden_layer_1_matmul_readvariableop_resource:G
9sequential_hidden_layer_1_biasadd_readvariableop_resource:J
8sequential_hidden_layer_2_matmul_readvariableop_resource:G
9sequential_hidden_layer_2_biasadd_readvariableop_resource:H
6sequential_output_layer_matmul_readvariableop_resource:E
7sequential_output_layer_biasadd_readvariableop_resource:
identity¢0sequential/Hidden-Layer-1/BiasAdd/ReadVariableOp¢/sequential/Hidden-Layer-1/MatMul/ReadVariableOp¢0sequential/Hidden-Layer-2/BiasAdd/ReadVariableOp¢/sequential/Hidden-Layer-2/MatMul/ReadVariableOp¢.sequential/Output-Layer/BiasAdd/ReadVariableOp¢-sequential/Output-Layer/MatMul/ReadVariableOp¨
/sequential/Hidden-Layer-1/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0«
 sequential/Hidden-Layer-1/MatMulMatMulhidden_layer_1_input7sequential/Hidden-Layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0sequential/Hidden-Layer-1/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ä
!sequential/Hidden-Layer-1/BiasAddBiasAdd*sequential/Hidden-Layer-1/MatMul:product:08sequential/Hidden-Layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential/Hidden-Layer-1/ReluRelu*sequential/Hidden-Layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¨
/sequential/Hidden-Layer-2/MatMul/ReadVariableOpReadVariableOp8sequential_hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0Ã
 sequential/Hidden-Layer-2/MatMulMatMul,sequential/Hidden-Layer-1/Relu:activations:07sequential/Hidden-Layer-2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¦
0sequential/Hidden-Layer-2/BiasAdd/ReadVariableOpReadVariableOp9sequential_hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ä
!sequential/Hidden-Layer-2/BiasAddBiasAdd*sequential/Hidden-Layer-2/MatMul:product:08sequential/Hidden-Layer-2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential/Hidden-Layer-2/ReluRelu*sequential/Hidden-Layer-2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¤
-sequential/Output-Layer/MatMul/ReadVariableOpReadVariableOp6sequential_output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¿
sequential/Output-Layer/MatMulMatMul,sequential/Hidden-Layer-2/Relu:activations:05sequential/Output-Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
.sequential/Output-Layer/BiasAdd/ReadVariableOpReadVariableOp7sequential_output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¾
sequential/Output-Layer/BiasAddBiasAdd(sequential/Output-Layer/MatMul:product:06sequential/Output-Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential/Output-Layer/SoftmaxSoftmax(sequential/Output-Layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿx
IdentityIdentity)sequential/Output-Layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿñ
NoOpNoOp1^sequential/Hidden-Layer-1/BiasAdd/ReadVariableOp0^sequential/Hidden-Layer-1/MatMul/ReadVariableOp1^sequential/Hidden-Layer-2/BiasAdd/ReadVariableOp0^sequential/Hidden-Layer-2/MatMul/ReadVariableOp/^sequential/Output-Layer/BiasAdd/ReadVariableOp.^sequential/Output-Layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2d
0sequential/Hidden-Layer-1/BiasAdd/ReadVariableOp0sequential/Hidden-Layer-1/BiasAdd/ReadVariableOp2b
/sequential/Hidden-Layer-1/MatMul/ReadVariableOp/sequential/Hidden-Layer-1/MatMul/ReadVariableOp2d
0sequential/Hidden-Layer-2/BiasAdd/ReadVariableOp0sequential/Hidden-Layer-2/BiasAdd/ReadVariableOp2b
/sequential/Hidden-Layer-2/MatMul/ReadVariableOp/sequential/Hidden-Layer-2/MatMul/ReadVariableOp2`
.sequential/Output-Layer/BiasAdd/ReadVariableOp.sequential/Output-Layer/BiasAdd/ReadVariableOp2^
-sequential/Output-Layer/MatMul/ReadVariableOp-sequential/Output-Layer/MatMul/ReadVariableOp:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_nameHidden-Layer-1_input
¢

÷
F__inference_Output-Layer_layer_call_and_return_conditional_losses_1903

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ù
H__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_1863

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê

)__inference_sequential_layer_call_fn_1793

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1668o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Æ

+__inference_Output-Layer_layer_call_fn_1892

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÛ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Output-Layer_layer_call_and_return_conditional_losses_1578o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

-__inference_Hidden-Layer-1_layer_call_fn_1852

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_1544o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ù
H__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_1544

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
è

"__inference_signature_wrapper_1759
hidden_layer_1_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallhidden_layer_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__wrapped_model_1526o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_nameHidden-Layer-1_input
¼
Ä
D__inference_sequential_layer_call_and_return_conditional_losses_1738
hidden_layer_1_input%
hidden_layer_1_1722:!
hidden_layer_1_1724:%
hidden_layer_2_1727:!
hidden_layer_2_1729:#
output_layer_1732:
output_layer_1734:
identity¢&Hidden-Layer-1/StatefulPartitionedCall¢&Hidden-Layer-2/StatefulPartitionedCall¢$Output-Layer/StatefulPartitionedCall
&Hidden-Layer-1/StatefulPartitionedCallStatefulPartitionedCallhidden_layer_1_inputhidden_layer_1_1722hidden_layer_1_1724*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_1544«
&Hidden-Layer-2/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-1/StatefulPartitionedCall:output:0hidden_layer_2_1727hidden_layer_2_1729*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_1561£
$Output-Layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-2/StatefulPartitionedCall:output:0output_layer_1732output_layer_1734*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Output-Layer_layer_call_and_return_conditional_losses_1578|
IdentityIdentity-Output-Layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
NoOpNoOp'^Hidden-Layer-1/StatefulPartitionedCall'^Hidden-Layer-2/StatefulPartitionedCall%^Output-Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2P
&Hidden-Layer-1/StatefulPartitionedCall&Hidden-Layer-1/StatefulPartitionedCall2P
&Hidden-Layer-2/StatefulPartitionedCall&Hidden-Layer-2/StatefulPartitionedCall2L
$Output-Layer/StatefulPartitionedCall$Output-Layer/StatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_nameHidden-Layer-1_input
Ê

-__inference_Hidden-Layer-2_layer_call_fn_1872

inputs
unknown:
	unknown_0:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_1561o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


ù
H__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_1561

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ê

)__inference_sequential_layer_call_fn_1776

inputs
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1585o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Å
D__inference_sequential_layer_call_and_return_conditional_losses_1843

inputs?
-hidden_layer_1_matmul_readvariableop_resource:<
.hidden_layer_1_biasadd_readvariableop_resource:?
-hidden_layer_2_matmul_readvariableop_resource:<
.hidden_layer_2_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:
identity¢%Hidden-Layer-1/BiasAdd/ReadVariableOp¢$Hidden-Layer-1/MatMul/ReadVariableOp¢%Hidden-Layer-2/BiasAdd/ReadVariableOp¢$Hidden-Layer-2/MatMul/ReadVariableOp¢#Output-Layer/BiasAdd/ReadVariableOp¢"Output-Layer/MatMul/ReadVariableOp
$Hidden-Layer-1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
Hidden-Layer-1/MatMulMatMulinputs,Hidden-Layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%Hidden-Layer-1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
Hidden-Layer-1/BiasAddBiasAddHidden-Layer-1/MatMul:product:0-Hidden-Layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
Hidden-Layer-1/ReluReluHidden-Layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$Hidden-Layer-2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¢
Hidden-Layer-2/MatMulMatMul!Hidden-Layer-1/Relu:activations:0,Hidden-Layer-2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%Hidden-Layer-2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
Hidden-Layer-2/BiasAddBiasAddHidden-Layer-2/MatMul:product:0-Hidden-Layer-2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
Hidden-Layer-2/ReluReluHidden-Layer-2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Output-Layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
Output-Layer/MatMulMatMul!Hidden-Layer-2/Relu:activations:0*Output-Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Output-Layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Output-Layer/BiasAddBiasAddOutput-Layer/MatMul:product:0+Output-Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
Output-Layer/SoftmaxSoftmaxOutput-Layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
IdentityIdentityOutput-Layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
NoOpNoOp&^Hidden-Layer-1/BiasAdd/ReadVariableOp%^Hidden-Layer-1/MatMul/ReadVariableOp&^Hidden-Layer-2/BiasAdd/ReadVariableOp%^Hidden-Layer-2/MatMul/ReadVariableOp$^Output-Layer/BiasAdd/ReadVariableOp#^Output-Layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2N
%Hidden-Layer-1/BiasAdd/ReadVariableOp%Hidden-Layer-1/BiasAdd/ReadVariableOp2L
$Hidden-Layer-1/MatMul/ReadVariableOp$Hidden-Layer-1/MatMul/ReadVariableOp2N
%Hidden-Layer-2/BiasAdd/ReadVariableOp%Hidden-Layer-2/BiasAdd/ReadVariableOp2L
$Hidden-Layer-2/MatMul/ReadVariableOp$Hidden-Layer-2/MatMul/ReadVariableOp2J
#Output-Layer/BiasAdd/ReadVariableOp#Output-Layer/BiasAdd/ReadVariableOp2H
"Output-Layer/MatMul/ReadVariableOp"Output-Layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Å
D__inference_sequential_layer_call_and_return_conditional_losses_1818

inputs?
-hidden_layer_1_matmul_readvariableop_resource:<
.hidden_layer_1_biasadd_readvariableop_resource:?
-hidden_layer_2_matmul_readvariableop_resource:<
.hidden_layer_2_biasadd_readvariableop_resource:=
+output_layer_matmul_readvariableop_resource::
,output_layer_biasadd_readvariableop_resource:
identity¢%Hidden-Layer-1/BiasAdd/ReadVariableOp¢$Hidden-Layer-1/MatMul/ReadVariableOp¢%Hidden-Layer-2/BiasAdd/ReadVariableOp¢$Hidden-Layer-2/MatMul/ReadVariableOp¢#Output-Layer/BiasAdd/ReadVariableOp¢"Output-Layer/MatMul/ReadVariableOp
$Hidden-Layer-1/MatMul/ReadVariableOpReadVariableOp-hidden_layer_1_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
Hidden-Layer-1/MatMulMatMulinputs,Hidden-Layer-1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%Hidden-Layer-1/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
Hidden-Layer-1/BiasAddBiasAddHidden-Layer-1/MatMul:product:0-Hidden-Layer-1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
Hidden-Layer-1/ReluReluHidden-Layer-1/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
$Hidden-Layer-2/MatMul/ReadVariableOpReadVariableOp-hidden_layer_2_matmul_readvariableop_resource*
_output_shapes

:*
dtype0¢
Hidden-Layer-2/MatMulMatMul!Hidden-Layer-1/Relu:activations:0,Hidden-Layer-2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
%Hidden-Layer-2/BiasAdd/ReadVariableOpReadVariableOp.hidden_layer_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0£
Hidden-Layer-2/BiasAddBiasAddHidden-Layer-2/MatMul:product:0-Hidden-Layer-2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿn
Hidden-Layer-2/ReluReluHidden-Layer-2/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"Output-Layer/MatMul/ReadVariableOpReadVariableOp+output_layer_matmul_readvariableop_resource*
_output_shapes

:*
dtype0
Output-Layer/MatMulMatMul!Hidden-Layer-2/Relu:activations:0*Output-Layer/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
#Output-Layer/BiasAdd/ReadVariableOpReadVariableOp,output_layer_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
Output-Layer/BiasAddBiasAddOutput-Layer/MatMul:product:0+Output-Layer/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿp
Output-Layer/SoftmaxSoftmaxOutput-Layer/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
IdentityIdentityOutput-Layer/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
NoOpNoOp&^Hidden-Layer-1/BiasAdd/ReadVariableOp%^Hidden-Layer-1/MatMul/ReadVariableOp&^Hidden-Layer-2/BiasAdd/ReadVariableOp%^Hidden-Layer-2/MatMul/ReadVariableOp$^Output-Layer/BiasAdd/ReadVariableOp#^Output-Layer/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2N
%Hidden-Layer-1/BiasAdd/ReadVariableOp%Hidden-Layer-1/BiasAdd/ReadVariableOp2L
$Hidden-Layer-1/MatMul/ReadVariableOp$Hidden-Layer-1/MatMul/ReadVariableOp2N
%Hidden-Layer-2/BiasAdd/ReadVariableOp%Hidden-Layer-2/BiasAdd/ReadVariableOp2L
$Hidden-Layer-2/MatMul/ReadVariableOp$Hidden-Layer-2/MatMul/ReadVariableOp2J
#Output-Layer/BiasAdd/ReadVariableOp#Output-Layer/BiasAdd/ReadVariableOp2H
"Output-Layer/MatMul/ReadVariableOp"Output-Layer/MatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	

)__inference_sequential_layer_call_fn_1700
hidden_layer_1_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallhidden_layer_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1668o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_nameHidden-Layer-1_input
	

)__inference_sequential_layer_call_fn_1600
hidden_layer_1_input
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallhidden_layer_1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *M
fHRF
D__inference_sequential_layer_call_and_return_conditional_losses_1585o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_nameHidden-Layer-1_input
÷-
¸
__inference__traced_save_1980
file_prefix4
0savev2_hidden_layer_1_kernel_read_readvariableop2
.savev2_hidden_layer_1_bias_read_readvariableop4
0savev2_hidden_layer_2_kernel_read_readvariableop2
.savev2_hidden_layer_2_bias_read_readvariableop2
.savev2_output_layer_kernel_read_readvariableop0
,savev2_output_layer_bias_read_readvariableop(
$savev2_iteration_read_readvariableop	,
(savev2_learning_rate_read_readvariableopE
Asavev2_rmsprop_velocity_hidden_layer_1_kernel_read_readvariableopC
?savev2_rmsprop_velocity_hidden_layer_1_bias_read_readvariableopE
Asavev2_rmsprop_velocity_hidden_layer_2_kernel_read_readvariableopC
?savev2_rmsprop_velocity_hidden_layer_2_bias_read_readvariableopC
?savev2_rmsprop_velocity_output_layer_kernel_read_readvariableopA
=savev2_rmsprop_velocity_output_layer_bias_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Å
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*î
valueäBáB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B ð
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_hidden_layer_1_kernel_read_readvariableop.savev2_hidden_layer_1_bias_read_readvariableop0savev2_hidden_layer_2_kernel_read_readvariableop.savev2_hidden_layer_2_bias_read_readvariableop.savev2_output_layer_kernel_read_readvariableop,savev2_output_layer_bias_read_readvariableop$savev2_iteration_read_readvariableop(savev2_learning_rate_read_readvariableopAsavev2_rmsprop_velocity_hidden_layer_1_kernel_read_readvariableop?savev2_rmsprop_velocity_hidden_layer_1_bias_read_readvariableopAsavev2_rmsprop_velocity_hidden_layer_2_kernel_read_readvariableop?savev2_rmsprop_velocity_hidden_layer_2_bias_read_readvariableop?savev2_rmsprop_velocity_output_layer_kernel_read_readvariableop=savev2_rmsprop_velocity_output_layer_bias_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopsavev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *!
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:³
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*
_input_shapesr
p: ::::::: : ::::::: : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$	 

_output_shapes

:: 


_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 


ù
H__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_1883

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿP
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿa
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¢

÷
F__inference_Output-Layer_layer_call_and_return_conditional_losses_1578

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
Ä
D__inference_sequential_layer_call_and_return_conditional_losses_1719
hidden_layer_1_input%
hidden_layer_1_1703:!
hidden_layer_1_1705:%
hidden_layer_2_1708:!
hidden_layer_2_1710:#
output_layer_1713:
output_layer_1715:
identity¢&Hidden-Layer-1/StatefulPartitionedCall¢&Hidden-Layer-2/StatefulPartitionedCall¢$Output-Layer/StatefulPartitionedCall
&Hidden-Layer-1/StatefulPartitionedCallStatefulPartitionedCallhidden_layer_1_inputhidden_layer_1_1703hidden_layer_1_1705*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_1544«
&Hidden-Layer-2/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-1/StatefulPartitionedCall:output:0hidden_layer_2_1708hidden_layer_2_1710*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_1561£
$Output-Layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-2/StatefulPartitionedCall:output:0output_layer_1713output_layer_1715*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Output-Layer_layer_call_and_return_conditional_losses_1578|
IdentityIdentity-Output-Layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
NoOpNoOp'^Hidden-Layer-1/StatefulPartitionedCall'^Hidden-Layer-2/StatefulPartitionedCall%^Output-Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2P
&Hidden-Layer-1/StatefulPartitionedCall&Hidden-Layer-1/StatefulPartitionedCall2P
&Hidden-Layer-2/StatefulPartitionedCall&Hidden-Layer-2/StatefulPartitionedCall2L
$Output-Layer/StatefulPartitionedCall$Output-Layer/StatefulPartitionedCall:] Y
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
.
_user_specified_nameHidden-Layer-1_input

¶
D__inference_sequential_layer_call_and_return_conditional_losses_1668

inputs%
hidden_layer_1_1652:!
hidden_layer_1_1654:%
hidden_layer_2_1657:!
hidden_layer_2_1659:#
output_layer_1662:
output_layer_1664:
identity¢&Hidden-Layer-1/StatefulPartitionedCall¢&Hidden-Layer-2/StatefulPartitionedCall¢$Output-Layer/StatefulPartitionedCall
&Hidden-Layer-1/StatefulPartitionedCallStatefulPartitionedCallinputshidden_layer_1_1652hidden_layer_1_1654*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_1544«
&Hidden-Layer-2/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-1/StatefulPartitionedCall:output:0hidden_layer_2_1657hidden_layer_2_1659*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_1561£
$Output-Layer/StatefulPartitionedCallStatefulPartitionedCall/Hidden-Layer-2/StatefulPartitionedCall:output:0output_layer_1662output_layer_1664*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *O
fJRH
F__inference_Output-Layer_layer_call_and_return_conditional_losses_1578|
IdentityIdentity-Output-Layer/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¿
NoOpNoOp'^Hidden-Layer-1/StatefulPartitionedCall'^Hidden-Layer-2/StatefulPartitionedCall%^Output-Layer/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ: : : : : : 2P
&Hidden-Layer-1/StatefulPartitionedCall&Hidden-Layer-1/StatefulPartitionedCall2P
&Hidden-Layer-2/StatefulPartitionedCall&Hidden-Layer-2/StatefulPartitionedCall2L
$Output-Layer/StatefulPartitionedCall$Output-Layer/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
P
¿
 __inference__traced_restore_2044
file_prefix8
&assignvariableop_hidden_layer_1_kernel:4
&assignvariableop_1_hidden_layer_1_bias::
(assignvariableop_2_hidden_layer_2_kernel:4
&assignvariableop_3_hidden_layer_2_bias:8
&assignvariableop_4_output_layer_kernel:2
$assignvariableop_5_output_layer_bias:&
assignvariableop_6_iteration:	 *
 assignvariableop_7_learning_rate: K
9assignvariableop_8_rmsprop_velocity_hidden_layer_1_kernel:E
7assignvariableop_9_rmsprop_velocity_hidden_layer_1_bias:L
:assignvariableop_10_rmsprop_velocity_hidden_layer_2_kernel:F
8assignvariableop_11_rmsprop_velocity_hidden_layer_2_bias:J
8assignvariableop_12_rmsprop_velocity_output_layer_kernel:D
6assignvariableop_13_rmsprop_velocity_output_layer_bias:%
assignvariableop_14_total_1: %
assignvariableop_15_count_1: #
assignvariableop_16_total: #
assignvariableop_17_count: 
identity_19¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_2¢AssignVariableOp_3¢AssignVariableOp_4¢AssignVariableOp_5¢AssignVariableOp_6¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9È
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*î
valueäBáB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB0optimizer/_iterations/.ATTRIBUTES/VARIABLE_VALUEB3optimizer/_learning_rate/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/1/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/2/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/3/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/4/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/5/.ATTRIBUTES/VARIABLE_VALUEB1optimizer/_variables/6/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*9
value0B.B B B B B B B B B B B B B B B B B B B ý
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*`
_output_shapesN
L:::::::::::::::::::*!
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:¹
AssignVariableOpAssignVariableOp&assignvariableop_hidden_layer_1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_1AssignVariableOp&assignvariableop_1_hidden_layer_1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:¿
AssignVariableOp_2AssignVariableOp(assignvariableop_2_hidden_layer_2_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_3AssignVariableOp&assignvariableop_3_hidden_layer_2_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:½
AssignVariableOp_4AssignVariableOp&assignvariableop_4_output_layer_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:»
AssignVariableOp_5AssignVariableOp$assignvariableop_5_output_layer_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:³
AssignVariableOp_6AssignVariableOpassignvariableop_6_iterationIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:·
AssignVariableOp_7AssignVariableOp assignvariableop_7_learning_rateIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:Ð
AssignVariableOp_8AssignVariableOp9assignvariableop_8_rmsprop_velocity_hidden_layer_1_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Î
AssignVariableOp_9AssignVariableOp7assignvariableop_9_rmsprop_velocity_hidden_layer_1_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:Ó
AssignVariableOp_10AssignVariableOp:assignvariableop_10_rmsprop_velocity_hidden_layer_2_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:Ñ
AssignVariableOp_11AssignVariableOp8assignvariableop_11_rmsprop_velocity_hidden_layer_2_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:Ñ
AssignVariableOp_12AssignVariableOp8assignvariableop_12_rmsprop_velocity_output_layer_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:Ï
AssignVariableOp_13AssignVariableOp6assignvariableop_13_rmsprop_velocity_output_layer_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_14AssignVariableOpassignvariableop_14_total_1Identity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:´
AssignVariableOp_15AssignVariableOpassignvariableop_15_count_1Identity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_16AssignVariableOpassignvariableop_16_totalIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:²
AssignVariableOp_17AssignVariableOpassignvariableop_17_countIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 Û
Identity_18Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_19IdentityIdentity_18:output:0^NoOp_1*
T0*
_output_shapes
: È
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_19Identity_19:output:0*9
_input_shapes(
&: : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix"
L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*É
serving_defaultµ
U
Hidden-Layer-1_input=
&serving_default_Hidden-Layer-1_input:0ÿÿÿÿÿÿÿÿÿ@
Output-Layer0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:ðq
Û
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*	&call_and_return_all_conditional_losses

_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias"
_tf_keras_layer
»
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses

#kernel
$bias"
_tf_keras_layer
J
0
1
2
3
#4
$5"
trackable_list_wrapper
J
0
1
2
3
#4
$5"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
%non_trainable_variables

&layers
'metrics
(layer_regularization_losses
)layer_metrics
	variables
trainable_variables
regularization_losses
__call__

_default_save_signature
*	&call_and_return_all_conditional_losses
&	"call_and_return_conditional_losses"
_generic_user_object
Ù
*trace_0
+trace_1
,trace_2
-trace_32î
)__inference_sequential_layer_call_fn_1600
)__inference_sequential_layer_call_fn_1776
)__inference_sequential_layer_call_fn_1793
)__inference_sequential_layer_call_fn_1700¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z*trace_0z+trace_1z,trace_2z-trace_3
Å
.trace_0
/trace_1
0trace_2
1trace_32Ú
D__inference_sequential_layer_call_and_return_conditional_losses_1818
D__inference_sequential_layer_call_and_return_conditional_losses_1843
D__inference_sequential_layer_call_and_return_conditional_losses_1719
D__inference_sequential_layer_call_and_return_conditional_losses_1738¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z.trace_0z/trace_1z0trace_2z1trace_3
×BÔ
__inference__wrapped_model_1526Hidden-Layer-1_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
´
2
_variables
3_iterations
4_learning_rate
5_index_dict
6_velocities
7
_momentums
8_average_gradients
9_update_step_xla"
experimentalOptimizer
,
:serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
;non_trainable_variables

<layers
=metrics
>layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ñ
@trace_02Ô
-__inference_Hidden-Layer-1_layer_call_fn_1852¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z@trace_0

Atrace_02ï
H__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_1863¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zAtrace_0
':%2Hidden-Layer-1/kernel
!:2Hidden-Layer-1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Bnon_trainable_variables

Clayers
Dmetrics
Elayer_regularization_losses
Flayer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ñ
Gtrace_02Ô
-__inference_Hidden-Layer-2_layer_call_fn_1872¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zGtrace_0

Htrace_02ï
H__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_1883¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zHtrace_0
':%2Hidden-Layer-2/kernel
!:2Hidden-Layer-2/bias
.
#0
$1"
trackable_list_wrapper
.
#0
$1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Inon_trainable_variables

Jlayers
Kmetrics
Llayer_regularization_losses
Mlayer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
ï
Ntrace_02Ò
+__inference_Output-Layer_layer_call_fn_1892¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zNtrace_0

Otrace_02í
F__inference_Output-Layer_layer_call_and_return_conditional_losses_1903¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zOtrace_0
%:#2Output-Layer/kernel
:2Output-Layer/bias
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
)__inference_sequential_layer_call_fn_1600Hidden-Layer-1_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
)__inference_sequential_layer_call_fn_1776inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
)__inference_sequential_layer_call_fn_1793inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
)__inference_sequential_layer_call_fn_1700Hidden-Layer-1_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
D__inference_sequential_layer_call_and_return_conditional_losses_1818inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
D__inference_sequential_layer_call_and_return_conditional_losses_1843inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
£B 
D__inference_sequential_layer_call_and_return_conditional_losses_1719Hidden-Layer-1_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
£B 
D__inference_sequential_layer_call_and_return_conditional_losses_1738Hidden-Layer-1_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Q
30
R1
S2
T3
U4
V5
W6"
trackable_list_wrapper
:	 2	iteration
: 2learning_rate
 "
trackable_dict_wrapper
J
R0
S1
T2
U3
V4
W5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¿2¼¹
®²ª
FullArgSpec2
args*'
jself

jgradient

jvariable
jkey
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
ÖBÓ
"__inference_signature_wrapper_1759Hidden-Layer-1_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
áBÞ
-__inference_Hidden-Layer-1_layer_call_fn_1852inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
üBù
H__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_1863inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
áBÞ
-__inference_Hidden-Layer-2_layer_call_fn_1872inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
üBù
H__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_1883inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ßBÜ
+__inference_Output-Layer_layer_call_fn_1892inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_Output-Layer_layer_call_and_return_conditional_losses_1903inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
N
X	variables
Y	keras_api
	Ztotal
	[count"
_tf_keras_metric
^
\	variables
]	keras_api
	^total
	_count
`
_fn_kwargs"
_tf_keras_metric
6:42&RMSprop/velocity/Hidden-Layer-1/kernel
0:.2$RMSprop/velocity/Hidden-Layer-1/bias
6:42&RMSprop/velocity/Hidden-Layer-2/kernel
0:.2$RMSprop/velocity/Hidden-Layer-2/bias
4:22$RMSprop/velocity/Output-Layer/kernel
.:,2"RMSprop/velocity/Output-Layer/bias
.
Z0
[1"
trackable_list_wrapper
-
X	variables"
_generic_user_object
:  (2total
:  (2count
.
^0
_1"
trackable_list_wrapper
-
\	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper¯
H__inference_Hidden-Layer-1_layer_call_and_return_conditional_losses_1863c/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 
-__inference_Hidden-Layer-1_layer_call_fn_1852X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "!
unknownÿÿÿÿÿÿÿÿÿ¯
H__inference_Hidden-Layer-2_layer_call_and_return_conditional_losses_1883c/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 
-__inference_Hidden-Layer-2_layer_call_fn_1872X/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "!
unknownÿÿÿÿÿÿÿÿÿ­
F__inference_Output-Layer_layer_call_and_return_conditional_losses_1903c#$/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 
+__inference_Output-Layer_layer_call_fn_1892X#$/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "!
unknownÿÿÿÿÿÿÿÿÿ¨
__inference__wrapped_model_1526#$=¢:
3¢0
.+
Hidden-Layer-1_inputÿÿÿÿÿÿÿÿÿ
ª ";ª8
6
Output-Layer&#
output_layerÿÿÿÿÿÿÿÿÿÅ
D__inference_sequential_layer_call_and_return_conditional_losses_1719}#$E¢B
;¢8
.+
Hidden-Layer-1_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 Å
D__inference_sequential_layer_call_and_return_conditional_losses_1738}#$E¢B
;¢8
.+
Hidden-Layer-1_inputÿÿÿÿÿÿÿÿÿ
p

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 ·
D__inference_sequential_layer_call_and_return_conditional_losses_1818o#$7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 ·
D__inference_sequential_layer_call_and_return_conditional_losses_1843o#$7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª ",¢)
"
tensor_0ÿÿÿÿÿÿÿÿÿ
 
)__inference_sequential_layer_call_fn_1600r#$E¢B
;¢8
.+
Hidden-Layer-1_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_1700r#$E¢B
;¢8
.+
Hidden-Layer-1_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_1776d#$7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "!
unknownÿÿÿÿÿÿÿÿÿ
)__inference_sequential_layer_call_fn_1793d#$7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "!
unknownÿÿÿÿÿÿÿÿÿÃ
"__inference_signature_wrapper_1759#$U¢R
¢ 
KªH
F
Hidden-Layer-1_input.+
hidden_layer_1_inputÿÿÿÿÿÿÿÿÿ";ª8
6
Output-Layer&#
output_layerÿÿÿÿÿÿÿÿÿ