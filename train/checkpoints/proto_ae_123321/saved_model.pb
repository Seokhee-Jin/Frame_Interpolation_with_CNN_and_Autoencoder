¡¾"
¤ô
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

Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

À
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

;
Elu
features"T
activations"T"
Ttype:
2
.
Identity

input"T
output"T"	
Ttype
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
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
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
÷
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
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
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8Ú¿

Adam/conv2d_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/v
y
(Adam/conv2d_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv2d_2/kernel/v

*Adam/conv2d_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/v*&
_output_shapes
:	*
dtype0

Adam/conv2d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/conv2d_transpose_1/bias/v

2Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_1/bias/v*
_output_shapes
:	*
dtype0
¤
 Adam/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*1
shared_name" Adam/conv2d_transpose_1/kernel/v

4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/v*&
_output_shapes
:		*
dtype0

Adam/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nameAdam/conv2d_transpose/bias/v

0Adam/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/v*
_output_shapes
:	*
dtype0
 
Adam/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*/
shared_name Adam/conv2d_transpose/kernel/v

2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/v*&
_output_shapes
:		*
dtype0

Adam/conv2d_transpose_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/conv2d_transpose_3/bias/v

2Adam/conv2d_transpose_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_3/bias/v*
_output_shapes
:	*
dtype0
¤
 Adam/conv2d_transpose_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*1
shared_name" Adam/conv2d_transpose_3/kernel/v

4Adam/conv2d_transpose_3/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_3/kernel/v*&
_output_shapes
:		*
dtype0

Adam/conv2d_transpose_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/conv2d_transpose_2/bias/v

2Adam/conv2d_transpose_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_2/bias/v*
_output_shapes
:	*
dtype0
¤
 Adam/conv2d_transpose_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/conv2d_transpose_2/kernel/v

4Adam/conv2d_transpose_2/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_2/kernel/v*&
_output_shapes
:	*
dtype0

Adam/conv2d_transpose_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_5/bias/v

2Adam/conv2d_transpose_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_5/bias/v*
_output_shapes
:*
dtype0
¤
 Adam/conv2d_transpose_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_5/kernel/v

4Adam/conv2d_transpose_5/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_5/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_transpose_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_4/bias/v

2Adam/conv2d_transpose_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_4/bias/v*
_output_shapes
:*
dtype0
¤
 Adam/conv2d_transpose_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_4/kernel/v

4Adam/conv2d_transpose_4/kernel/v/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_4/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_6/bias/v
y
(Adam/conv2d_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_6/kernel/v

*Adam/conv2d_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/v
y
(Adam/conv2d_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_5/kernel/v

*Adam/conv2d_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/v
y
(Adam/conv2d_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/v

*Adam/conv2d_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/v*&
_output_shapes
:*
dtype0

Adam/conv2d_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/v
y
(Adam/conv2d_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/v*
_output_shapes
:*
dtype0

Adam/conv2d_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv2d_3/kernel/v

*Adam/conv2d_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/v*&
_output_shapes
:	*
dtype0

Adam/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/conv2d_1/bias/v
y
(Adam/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/v*
_output_shapes
:	*
dtype0

Adam/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*'
shared_nameAdam/conv2d_1/kernel/v

*Adam/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/v*&
_output_shapes
:		*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:	*
dtype0

Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/conv2d/kernel/v

(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:	*
dtype0

Adam/conv2d_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_2/bias/m
y
(Adam/conv2d_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv2d_2/kernel/m

*Adam/conv2d_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_2/kernel/m*&
_output_shapes
:	*
dtype0

Adam/conv2d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/conv2d_transpose_1/bias/m

2Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_1/bias/m*
_output_shapes
:	*
dtype0
¤
 Adam/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*1
shared_name" Adam/conv2d_transpose_1/kernel/m

4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_1/kernel/m*&
_output_shapes
:		*
dtype0

Adam/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*-
shared_nameAdam/conv2d_transpose/bias/m

0Adam/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/bias/m*
_output_shapes
:	*
dtype0
 
Adam/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*/
shared_name Adam/conv2d_transpose/kernel/m

2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose/kernel/m*&
_output_shapes
:		*
dtype0

Adam/conv2d_transpose_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/conv2d_transpose_3/bias/m

2Adam/conv2d_transpose_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_3/bias/m*
_output_shapes
:	*
dtype0
¤
 Adam/conv2d_transpose_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*1
shared_name" Adam/conv2d_transpose_3/kernel/m

4Adam/conv2d_transpose_3/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_3/kernel/m*&
_output_shapes
:		*
dtype0

Adam/conv2d_transpose_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*/
shared_name Adam/conv2d_transpose_2/bias/m

2Adam/conv2d_transpose_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_2/bias/m*
_output_shapes
:	*
dtype0
¤
 Adam/conv2d_transpose_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*1
shared_name" Adam/conv2d_transpose_2/kernel/m

4Adam/conv2d_transpose_2/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_2/kernel/m*&
_output_shapes
:	*
dtype0

Adam/conv2d_transpose_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_5/bias/m

2Adam/conv2d_transpose_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_5/bias/m*
_output_shapes
:*
dtype0
¤
 Adam/conv2d_transpose_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_5/kernel/m

4Adam/conv2d_transpose_5/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_5/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_transpose_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*/
shared_name Adam/conv2d_transpose_4/bias/m

2Adam/conv2d_transpose_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_transpose_4/bias/m*
_output_shapes
:*
dtype0
¤
 Adam/conv2d_transpose_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/conv2d_transpose_4/kernel/m

4Adam/conv2d_transpose_4/kernel/m/Read/ReadVariableOpReadVariableOp Adam/conv2d_transpose_4/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_6/bias/m
y
(Adam/conv2d_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_6/kernel/m

*Adam/conv2d_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_6/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_5/bias/m
y
(Adam/conv2d_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_5/kernel/m

*Adam/conv2d_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_5/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_4/bias/m
y
(Adam/conv2d_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_nameAdam/conv2d_4/kernel/m

*Adam/conv2d_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_4/kernel/m*&
_output_shapes
:*
dtype0

Adam/conv2d_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/conv2d_3/bias/m
y
(Adam/conv2d_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/bias/m*
_output_shapes
:*
dtype0

Adam/conv2d_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/conv2d_3/kernel/m

*Adam/conv2d_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_3/kernel/m*&
_output_shapes
:	*
dtype0

Adam/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/conv2d_1/bias/m
y
(Adam/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/bias/m*
_output_shapes
:	*
dtype0

Adam/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*'
shared_nameAdam/conv2d_1/kernel/m

*Adam/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_1/kernel/m*&
_output_shapes
:		*
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:	*
dtype0

Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*%
shared_nameAdam/conv2d/kernel/m

(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:	*
dtype0
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
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes
:*
dtype0

conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
:	*
dtype0

conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameconv2d_transpose_1/bias

+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes
:	*
dtype0

conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		**
shared_nameconv2d_transpose_1/kernel

-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*&
_output_shapes
:		*
dtype0

conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameconv2d_transpose/bias
{
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes
:	*
dtype0

conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		*(
shared_nameconv2d_transpose/kernel

+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*&
_output_shapes
:		*
dtype0

conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameconv2d_transpose_3/bias

+conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/bias*
_output_shapes
:	*
dtype0

conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		**
shared_nameconv2d_transpose_3/kernel

-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*&
_output_shapes
:		*
dtype0

conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*(
shared_nameconv2d_transpose_2/bias

+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
_output_shapes
:	*
dtype0

conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	**
shared_nameconv2d_transpose_2/kernel

-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*&
_output_shapes
:	*
dtype0

conv2d_transpose_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_5/bias

+conv2d_transpose_5/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/bias*
_output_shapes
:*
dtype0

conv2d_transpose_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_5/kernel

-conv2d_transpose_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_5/kernel*&
_output_shapes
:*
dtype0

conv2d_transpose_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_4/bias

+conv2d_transpose_4/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/bias*
_output_shapes
:*
dtype0

conv2d_transpose_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameconv2d_transpose_4/kernel

-conv2d_transpose_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_4/kernel*&
_output_shapes
:*
dtype0
r
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_6/bias
k
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes
:*
dtype0

conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_6/kernel
{
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*&
_output_shapes
:*
dtype0
r
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_5/bias
k
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes
:*
dtype0

conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_5/kernel
{
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*&
_output_shapes
:*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:*
dtype0

conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:*
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
:*
dtype0

conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
:	*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:	*
dtype0

conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:		* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:		*
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:	*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:	*
dtype0

 serving_default_proto_enc1_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿdd

StatefulPartitionedCallStatefulPartitionedCall serving_default_proto_enc1_inputconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_transpose_4/kernelconv2d_transpose_4/biasconv2d_transpose_5/kernelconv2d_transpose_5/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_2/kernelconv2d_2/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *-
f(R&
$__inference_signature_wrapper_551429

NoOpNoOp
Æ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*½Å
value²ÅB®Å B¦Å
¶
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories*

layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
##_self_saveable_object_factories*

$layer-0
%layer_with_weights-0
%layer-1
&layer_with_weights-1
&layer-2
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
#-_self_saveable_object_factories*

.layer-0
/layer_with_weights-0
/layer-1
0layer_with_weights-1
0layer-2
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
#7_self_saveable_object_factories*

8layer-0
9layer_with_weights-0
9layer-1
:layer_with_weights-1
:layer-2
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
#A_self_saveable_object_factories*
·
Blayer-0
Clayer_with_weights-0
Clayer-1
Dlayer_with_weights-1
Dlayer-2
Elayer_with_weights-2
Elayer-3
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
#L_self_saveable_object_factories*
Ê
M0
N1
O2
P3
Q4
R5
S6
T7
U8
V9
W10
X11
Y12
Z13
[14
\15
]16
^17
_18
`19
a20
b21
c22
d23
e24
f25*
Ê
M0
N1
O2
P3
Q4
R5
S6
T7
U8
V9
W10
X11
Y12
Z13
[14
\15
]16
^17
_18
`19
a20
b21
c22
d23
e24
f25*
* 
°
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
6
ltrace_0
mtrace_1
ntrace_2
otrace_3* 
6
ptrace_0
qtrace_1
rtrace_2
strace_3* 
* 
Ì
titer

ubeta_1

vbeta_2
	wdecay
xlearning_rateMmNmOmPmQm Rm¡Sm¢Tm£Um¤Vm¥Wm¦Xm§Ym¨Zm©[mª\m«]m¬^m­_m®`m¯am°bm±cm²dm³em´fmµMv¶Nv·Ov¸Pv¹QvºRv»Sv¼Tv½Uv¾Vv¿WvÀXvÁYvÂZvÃ[vÄ\vÅ]vÆ^vÇ_vÈ`vÉavÊbvËcvÌdvÍevÎfvÏ*

yserving_default* 
'
#z_self_saveable_object_factories* 
ð
{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+&call_and_return_all_conditional_losses

Mkernel
Nbias
$_self_saveable_object_factories
!_jit_compiled_convolution_op*
õ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

Okernel
Pbias
$_self_saveable_object_factories
!_jit_compiled_convolution_op*
 
M0
N1
O2
P3*
 
M0
N1
O2
P3*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
(
$_self_saveable_object_factories* 
õ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

Qkernel
Rbias
$_self_saveable_object_factories
! _jit_compiled_convolution_op*
õ
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses

Skernel
Tbias
$§_self_saveable_object_factories
!¨_jit_compiled_convolution_op*
 
Q0
R1
S2
T3*
 
Q0
R1
S2
T3*
* 

©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses*
:
®trace_0
¯trace_1
°trace_2
±trace_3* 
:
²trace_0
³trace_1
´trace_2
µtrace_3* 
* 
(
$¶_self_saveable_object_factories* 
õ
·	variables
¸trainable_variables
¹regularization_losses
º	keras_api
»__call__
+¼&call_and_return_all_conditional_losses

Ukernel
Vbias
$½_self_saveable_object_factories
!¾_jit_compiled_convolution_op*
õ
¿	variables
Àtrainable_variables
Áregularization_losses
Â	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses

Wkernel
Xbias
$Å_self_saveable_object_factories
!Æ_jit_compiled_convolution_op*
 
U0
V1
W2
X3*
 
U0
V1
W2
X3*
* 

Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses*
:
Ìtrace_0
Ítrace_1
Îtrace_2
Ïtrace_3* 
:
Ðtrace_0
Ñtrace_1
Òtrace_2
Ótrace_3* 
* 
(
$Ô_self_saveable_object_factories* 
õ
Õ	variables
Ötrainable_variables
×regularization_losses
Ø	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses

Ykernel
Zbias
$Û_self_saveable_object_factories
!Ü_jit_compiled_convolution_op*
õ
Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
á__call__
+â&call_and_return_all_conditional_losses

[kernel
\bias
$ã_self_saveable_object_factories
!ä_jit_compiled_convolution_op*
 
Y0
Z1
[2
\3*
 
Y0
Z1
[2
\3*
* 

ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses*
:
êtrace_0
ëtrace_1
ìtrace_2
ítrace_3* 
:
îtrace_0
ïtrace_1
ðtrace_2
ñtrace_3* 
* 
(
$ò_self_saveable_object_factories* 
õ
ó	variables
ôtrainable_variables
õregularization_losses
ö	keras_api
÷__call__
+ø&call_and_return_all_conditional_losses

]kernel
^bias
$ù_self_saveable_object_factories
!ú_jit_compiled_convolution_op*
õ
û	variables
ütrainable_variables
ýregularization_losses
þ	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses

_kernel
`bias
$_self_saveable_object_factories
!_jit_compiled_convolution_op*
 
]0
^1
_2
`3*
 
]0
^1
_2
`3*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
(
$_self_saveable_object_factories* 
õ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

akernel
bbias
$_self_saveable_object_factories
!_jit_compiled_convolution_op*
õ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

ckernel
dbias
$_self_saveable_object_factories
! _jit_compiled_convolution_op*
õ
¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses

ekernel
fbias
$§_self_saveable_object_factories
!¨_jit_compiled_convolution_op*
.
a0
b1
c2
d3
e4
f5*
.
a0
b1
c2
d3
e4
f5*
* 

©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses*
:
®trace_0
¯trace_1
°trace_2
±trace_3* 
:
²trace_0
³trace_1
´trace_2
µtrace_3* 
* 
MG
VARIABLE_VALUEconv2d/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUEconv2d/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_3/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_3/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_4/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_4/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEconv2d_5/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
MG
VARIABLE_VALUEconv2d_5/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_6/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_6/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_4/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_4/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_5/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_5/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_2/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_2/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_3/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_3/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose/kernel'variables/20/.ATTRIBUTES/VARIABLE_VALUE*
VP
VARIABLE_VALUEconv2d_transpose/bias'variables/21/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEconv2d_transpose_1/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUEconv2d_transpose_1/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEconv2d_2/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUEconv2d_2/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
* 
.
0
1
2
3
4
5*

¶0
·1*
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
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 

M0
N1*

M0
N1*
* 

¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

½trace_0* 

¾trace_0* 
* 
* 

O0
P1*

O0
P1*
* 

¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Ätrace_0* 

Åtrace_0* 
* 
* 
* 

0
1
2*
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

Q0
R1*

Q0
R1*
* 

Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Ëtrace_0* 

Ìtrace_0* 
* 
* 

S0
T1*

S0
T1*
* 

Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses*

Òtrace_0* 

Ótrace_0* 
* 
* 
* 

0
1
2*
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

U0
V1*

U0
V1*
* 

Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
·	variables
¸trainable_variables
¹regularization_losses
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses*

Ùtrace_0* 

Útrace_0* 
* 
* 

W0
X1*

W0
X1*
* 

Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
¿	variables
Àtrainable_variables
Áregularization_losses
Ã__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses*

àtrace_0* 

átrace_0* 
* 
* 
* 

$0
%1
&2*
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

Y0
Z1*

Y0
Z1*
* 

ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
Õ	variables
Ötrainable_variables
×regularization_losses
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses*

çtrace_0* 

ètrace_0* 
* 
* 

[0
\1*

[0
\1*
* 

énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
Ý	variables
Þtrainable_variables
ßregularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses*

îtrace_0* 

ïtrace_0* 
* 
* 
* 

.0
/1
02*
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

]0
^1*

]0
^1*
* 

ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
ó	variables
ôtrainable_variables
õregularization_losses
÷__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses*

õtrace_0* 

ötrace_0* 
* 
* 

_0
`1*

_0
`1*
* 

÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
û	variables
ütrainable_variables
ýregularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

ütrace_0* 

ýtrace_0* 
* 
* 
* 

80
91
:2*
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

a0
b1*

a0
b1*
* 

þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
* 

c0
d1*

c0
d1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
* 

e0
f1*

e0
f1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
* 
* 
 
B0
C1
D2
E3*
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
<
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count

_fn_kwargs*
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

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
pj
VARIABLE_VALUEAdam/conv2d/kernel/mBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv2d/bias/mBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/mBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_1/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_3/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_3/bias/mBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_4/kernel/mBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_4/bias/mBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_5/kernel/mBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_5/bias/mBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_6/kernel/mCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_6/bias/mCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_4/kernel/mCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_4/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_5/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_5/bias/mCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_2/kernel/mCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_2/bias/mCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_3/kernel/mCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_3/bias/mCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose/kernel/mCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/conv2d_transpose/bias/mCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/mCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_1/bias/mCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_2/kernel/mCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_2/bias/mCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d/kernel/vBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
nh
VARIABLE_VALUEAdam/conv2d/bias/vBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_1/kernel/vBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_1/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_3/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_3/bias/vBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_4/kernel/vBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_4/bias/vBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
rl
VARIABLE_VALUEAdam/conv2d_5/kernel/vBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
pj
VARIABLE_VALUEAdam/conv2d_5/bias/vBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_6/kernel/vCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_6/bias/vCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_4/kernel/vCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_4/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_5/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_5/bias/vCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_2/kernel/vCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_2/bias/vCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_3/kernel/vCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_3/bias/vCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose/kernel/vCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
ys
VARIABLE_VALUEAdam/conv2d_transpose/bias/vCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}w
VARIABLE_VALUE Adam/conv2d_transpose_1/kernel/vCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
{u
VARIABLE_VALUEAdam/conv2d_transpose_1/bias/vCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
sm
VARIABLE_VALUEAdam/conv2d_2/kernel/vCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
qk
VARIABLE_VALUEAdam/conv2d_2/bias/vCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ä 
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp-conv2d_transpose_4/kernel/Read/ReadVariableOp+conv2d_transpose_4/bias/Read/ReadVariableOp-conv2d_transpose_5/kernel/Read/ReadVariableOp+conv2d_transpose_5/bias/Read/ReadVariableOp-conv2d_transpose_2/kernel/Read/ReadVariableOp+conv2d_transpose_2/bias/Read/ReadVariableOp-conv2d_transpose_3/kernel/Read/ReadVariableOp+conv2d_transpose_3/bias/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp*Adam/conv2d_1/kernel/m/Read/ReadVariableOp(Adam/conv2d_1/bias/m/Read/ReadVariableOp*Adam/conv2d_3/kernel/m/Read/ReadVariableOp(Adam/conv2d_3/bias/m/Read/ReadVariableOp*Adam/conv2d_4/kernel/m/Read/ReadVariableOp(Adam/conv2d_4/bias/m/Read/ReadVariableOp*Adam/conv2d_5/kernel/m/Read/ReadVariableOp(Adam/conv2d_5/bias/m/Read/ReadVariableOp*Adam/conv2d_6/kernel/m/Read/ReadVariableOp(Adam/conv2d_6/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_4/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_4/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_5/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_5/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_2/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_2/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_3/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_3/bias/m/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/m/Read/ReadVariableOp0Adam/conv2d_transpose/bias/m/Read/ReadVariableOp4Adam/conv2d_transpose_1/kernel/m/Read/ReadVariableOp2Adam/conv2d_transpose_1/bias/m/Read/ReadVariableOp*Adam/conv2d_2/kernel/m/Read/ReadVariableOp(Adam/conv2d_2/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp*Adam/conv2d_1/kernel/v/Read/ReadVariableOp(Adam/conv2d_1/bias/v/Read/ReadVariableOp*Adam/conv2d_3/kernel/v/Read/ReadVariableOp(Adam/conv2d_3/bias/v/Read/ReadVariableOp*Adam/conv2d_4/kernel/v/Read/ReadVariableOp(Adam/conv2d_4/bias/v/Read/ReadVariableOp*Adam/conv2d_5/kernel/v/Read/ReadVariableOp(Adam/conv2d_5/bias/v/Read/ReadVariableOp*Adam/conv2d_6/kernel/v/Read/ReadVariableOp(Adam/conv2d_6/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_4/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_4/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_5/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_5/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_2/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_2/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_3/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_3/bias/v/Read/ReadVariableOp2Adam/conv2d_transpose/kernel/v/Read/ReadVariableOp0Adam/conv2d_transpose/bias/v/Read/ReadVariableOp4Adam/conv2d_transpose_1/kernel/v/Read/ReadVariableOp2Adam/conv2d_transpose_1/bias/v/Read/ReadVariableOp*Adam/conv2d_2/kernel/v/Read/ReadVariableOp(Adam/conv2d_2/bias/v/Read/ReadVariableOpConst*d
Tin]
[2Y	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *(
f#R!
__inference__traced_save_553116
ó
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/biasconv2d_6/kernelconv2d_6/biasconv2d_transpose_4/kernelconv2d_transpose_4/biasconv2d_transpose_5/kernelconv2d_transpose_5/biasconv2d_transpose_2/kernelconv2d_transpose_2/biasconv2d_transpose_3/kernelconv2d_transpose_3/biasconv2d_transpose/kernelconv2d_transpose/biasconv2d_transpose_1/kernelconv2d_transpose_1/biasconv2d_2/kernelconv2d_2/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d/kernel/mAdam/conv2d/bias/mAdam/conv2d_1/kernel/mAdam/conv2d_1/bias/mAdam/conv2d_3/kernel/mAdam/conv2d_3/bias/mAdam/conv2d_4/kernel/mAdam/conv2d_4/bias/mAdam/conv2d_5/kernel/mAdam/conv2d_5/bias/mAdam/conv2d_6/kernel/mAdam/conv2d_6/bias/m Adam/conv2d_transpose_4/kernel/mAdam/conv2d_transpose_4/bias/m Adam/conv2d_transpose_5/kernel/mAdam/conv2d_transpose_5/bias/m Adam/conv2d_transpose_2/kernel/mAdam/conv2d_transpose_2/bias/m Adam/conv2d_transpose_3/kernel/mAdam/conv2d_transpose_3/bias/mAdam/conv2d_transpose/kernel/mAdam/conv2d_transpose/bias/m Adam/conv2d_transpose_1/kernel/mAdam/conv2d_transpose_1/bias/mAdam/conv2d_2/kernel/mAdam/conv2d_2/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/conv2d_1/kernel/vAdam/conv2d_1/bias/vAdam/conv2d_3/kernel/vAdam/conv2d_3/bias/vAdam/conv2d_4/kernel/vAdam/conv2d_4/bias/vAdam/conv2d_5/kernel/vAdam/conv2d_5/bias/vAdam/conv2d_6/kernel/vAdam/conv2d_6/bias/v Adam/conv2d_transpose_4/kernel/vAdam/conv2d_transpose_4/bias/v Adam/conv2d_transpose_5/kernel/vAdam/conv2d_transpose_5/bias/v Adam/conv2d_transpose_2/kernel/vAdam/conv2d_transpose_2/bias/v Adam/conv2d_transpose_3/kernel/vAdam/conv2d_transpose_3/bias/vAdam/conv2d_transpose/kernel/vAdam/conv2d_transpose/bias/v Adam/conv2d_transpose_1/kernel/vAdam/conv2d_transpose_1/bias/vAdam/conv2d_2/kernel/vAdam/conv2d_2/bias/v*c
Tin\
Z2X*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2 *0J 8 *+
f&R$
"__inference__traced_restore_553387 
®	

+__inference_proto_dec1_layer_call_fn_550848
input_2!
unknown:		
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:	
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550816w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ22	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
!
_user_specified_name	input_2
Ï
Þ
+__inference_proto_dec2_layer_call_fn_552213

inputs!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550577w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
¾)
¾

K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_550952

inputs+
proto_enc1_550893:	
proto_enc1_550895:	+
proto_enc1_550897:		
proto_enc1_550899:	+
proto_enc2_550902:	
proto_enc2_550904:+
proto_enc2_550906:
proto_enc2_550908:+
proto_enc3_550911:
proto_enc3_550913:+
proto_enc3_550915:
proto_enc3_550917:+
proto_dec3_550920:
proto_dec3_550922:+
proto_dec3_550924:
proto_dec3_550926:+
proto_dec2_550929:	
proto_dec2_550931:	+
proto_dec2_550933:		
proto_dec2_550935:	+
proto_dec1_550938:		
proto_dec1_550940:	+
proto_dec1_550942:		
proto_dec1_550944:	+
proto_dec1_550946:	
proto_dec1_550948:
identity¢"proto_dec1/StatefulPartitionedCall¢"proto_dec2/StatefulPartitionedCall¢"proto_dec3/StatefulPartitionedCall¢"proto_enc1/StatefulPartitionedCall¢"proto_enc2/StatefulPartitionedCall¢"proto_enc3/StatefulPartitionedCall¯
"proto_enc1/StatefulPartitionedCallStatefulPartitionedCallinputsproto_enc1_550893proto_enc1_550895proto_enc1_550897proto_enc1_550899*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549809Ô
"proto_enc2/StatefulPartitionedCallStatefulPartitionedCall+proto_enc1/StatefulPartitionedCall:output:0proto_enc2_550902proto_enc2_550904proto_enc2_550906proto_enc2_550908*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc2_layer_call_and_return_conditional_losses_549963Ô
"proto_enc3/StatefulPartitionedCallStatefulPartitionedCall+proto_enc2/StatefulPartitionedCall:output:0proto_enc3_550911proto_enc3_550913proto_enc3_550915proto_enc3_550917*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550117Ô
"proto_dec3/StatefulPartitionedCallStatefulPartitionedCall+proto_enc3/StatefulPartitionedCall:output:0proto_dec3_550920proto_dec3_550922proto_dec3_550924proto_dec3_550926*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550337Ô
"proto_dec2/StatefulPartitionedCallStatefulPartitionedCall+proto_dec3/StatefulPartitionedCall:output:0proto_dec2_550929proto_dec2_550931proto_dec2_550933proto_dec2_550935*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550537þ
"proto_dec1/StatefulPartitionedCallStatefulPartitionedCall+proto_dec2/StatefulPartitionedCall:output:0proto_dec1_550938proto_dec1_550940proto_dec1_550942proto_dec1_550944proto_dec1_550946proto_dec1_550948*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550753
IdentityIdentity+proto_dec1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¤
NoOpNoOp#^proto_dec1/StatefulPartitionedCall#^proto_dec2/StatefulPartitionedCall#^proto_dec3/StatefulPartitionedCall#^proto_enc1/StatefulPartitionedCall#^proto_enc2/StatefulPartitionedCall#^proto_enc3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿdd: : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"proto_dec1/StatefulPartitionedCall"proto_dec1/StatefulPartitionedCall2H
"proto_dec2/StatefulPartitionedCall"proto_dec2/StatefulPartitionedCall2H
"proto_dec3/StatefulPartitionedCall"proto_dec3/StatefulPartitionedCall2H
"proto_enc1/StatefulPartitionedCall"proto_enc1/StatefulPartitionedCall2H
"proto_enc2/StatefulPartitionedCall"proto_enc2/StatefulPartitionedCall2H
"proto_enc3/StatefulPartitionedCall"proto_enc3/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
¾)
¾

K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551128

inputs+
proto_enc1_551069:	
proto_enc1_551071:	+
proto_enc1_551073:		
proto_enc1_551075:	+
proto_enc2_551078:	
proto_enc2_551080:+
proto_enc2_551082:
proto_enc2_551084:+
proto_enc3_551087:
proto_enc3_551089:+
proto_enc3_551091:
proto_enc3_551093:+
proto_dec3_551096:
proto_dec3_551098:+
proto_dec3_551100:
proto_dec3_551102:+
proto_dec2_551105:	
proto_dec2_551107:	+
proto_dec2_551109:		
proto_dec2_551111:	+
proto_dec1_551114:		
proto_dec1_551116:	+
proto_dec1_551118:		
proto_dec1_551120:	+
proto_dec1_551122:	
proto_dec1_551124:
identity¢"proto_dec1/StatefulPartitionedCall¢"proto_dec2/StatefulPartitionedCall¢"proto_dec3/StatefulPartitionedCall¢"proto_enc1/StatefulPartitionedCall¢"proto_enc2/StatefulPartitionedCall¢"proto_enc3/StatefulPartitionedCall¯
"proto_enc1/StatefulPartitionedCallStatefulPartitionedCallinputsproto_enc1_551069proto_enc1_551071proto_enc1_551073proto_enc1_551075*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549869Ô
"proto_enc2/StatefulPartitionedCallStatefulPartitionedCall+proto_enc1/StatefulPartitionedCall:output:0proto_enc2_551078proto_enc2_551080proto_enc2_551082proto_enc2_551084*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc2_layer_call_and_return_conditional_losses_550023Ô
"proto_enc3/StatefulPartitionedCallStatefulPartitionedCall+proto_enc2/StatefulPartitionedCall:output:0proto_enc3_551087proto_enc3_551089proto_enc3_551091proto_enc3_551093*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550177Ô
"proto_dec3/StatefulPartitionedCallStatefulPartitionedCall+proto_enc3/StatefulPartitionedCall:output:0proto_dec3_551096proto_dec3_551098proto_dec3_551100proto_dec3_551102*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550377Ô
"proto_dec2/StatefulPartitionedCallStatefulPartitionedCall+proto_dec3/StatefulPartitionedCall:output:0proto_dec2_551105proto_dec2_551107proto_dec2_551109proto_dec2_551111*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550577þ
"proto_dec1/StatefulPartitionedCallStatefulPartitionedCall+proto_dec2/StatefulPartitionedCall:output:0proto_dec1_551114proto_dec1_551116proto_dec1_551118proto_dec1_551120proto_dec1_551122proto_dec1_551124*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550816
IdentityIdentity+proto_dec1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¤
NoOpNoOp#^proto_dec1/StatefulPartitionedCall#^proto_dec2/StatefulPartitionedCall#^proto_dec3/StatefulPartitionedCall#^proto_enc1/StatefulPartitionedCall#^proto_enc2/StatefulPartitionedCall#^proto_enc3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿdd: : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"proto_dec1/StatefulPartitionedCall"proto_dec1/StatefulPartitionedCall2H
"proto_dec2/StatefulPartitionedCall"proto_dec2/StatefulPartitionedCall2H
"proto_dec3/StatefulPartitionedCall"proto_dec3/StatefulPartitionedCall2H
"proto_enc1/StatefulPartitionedCall"proto_enc1/StatefulPartitionedCall2H
"proto_enc2/StatefulPartitionedCall"proto_enc2/StatefulPartitionedCall2H
"proto_enc3/StatefulPartitionedCall"proto_enc3/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
ï

)__inference_conv2d_4_layer_call_fn_552504

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_549956w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
®9
Ù
F__inference_proto_dec3_layer_call_and_return_conditional_losses_552187

inputsU
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_4_biasadd_readvariableop_resource:U
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_5_biasadd_readvariableop_resource:
identity¢)conv2d_transpose_4/BiasAdd/ReadVariableOp¢2conv2d_transpose_4/conv2d_transpose/ReadVariableOp¢)conv2d_transpose_5/BiasAdd/ReadVariableOp¢2conv2d_transpose_5/conv2d_transpose/ReadVariableOpN
conv2d_transpose_4/ShapeShapeinputs*
T0*
_output_shapes
:p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :è
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¶
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0À
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22|
conv2d_transpose_4/EluElu#conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22l
conv2d_transpose_5/ShapeShape$conv2d_transpose_4/Elu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :è
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¶
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0¡
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0$conv2d_transpose_4/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0À
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22|
conv2d_transpose_5/EluElu#conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22{
IdentityIdentity$conv2d_transpose_5/Elu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
NoOpNoOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
£
ã
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550867
input_21
conv2d_transpose_550851:		%
conv2d_transpose_550853:	3
conv2d_transpose_1_550856:		'
conv2d_transpose_1_550858:	)
conv2d_2_550861:	
conv2d_2_550863:
identity¢ conv2d_2/StatefulPartitionedCall¢(conv2d_transpose/StatefulPartitionedCall¢*conv2d_transpose_1/StatefulPartitionedCall
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_transpose_550851conv2d_transpose_550853*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_550667Ð
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_550856conv2d_transpose_1_550858*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_550712ª
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_2_550861conv2d_2_550863*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_550746
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÁ
NoOpNoOp!^conv2d_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ22	: : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
!
_user_specified_name	input_2
Á!

N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_550512

inputsB
(conv2d_transpose_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :	y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs

ý
D__inference_conv2d_1_layer_call_and_return_conditional_losses_552475

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs
®	

+__inference_proto_dec1_layer_call_fn_550768
input_2!
unknown:		
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:	
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550753w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ22	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
!
_user_specified_name	input_2

ý
D__inference_conv2d_1_layer_call_and_return_conditional_losses_549802

inputs8
conv2d_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs
Ï
Þ
+__inference_proto_dec3_layer_call_fn_552086

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550337w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Á!

N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_550467

inputsB
(conv2d_transpose_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :	y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:	*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ý
D__inference_conv2d_5_layer_call_and_return_conditional_losses_550093

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
à
À
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550229
input_5)
conv2d_5_550218:
conv2d_5_550220:)
conv2d_6_550223:
conv2d_6_550225:
identity¢ conv2d_5/StatefulPartitionedCall¢ conv2d_6/StatefulPartitionedCallþ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_5_550218conv2d_5_550220*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_550093 
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_550223conv2d_6_550225*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_550110
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
NoOpNoOp!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input_5
Ò
ß
+__inference_proto_enc3_layer_call_fn_550128
input_5!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550117w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input_5
Ì
¨
3__inference_conv2d_transpose_5_layer_call_fn_552607

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_550312
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á!

N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_552813

inputsB
(conv2d_transpose_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :	y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
Á!

N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_550267

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿh
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿz
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ï

)__inference_conv2d_2_layer_call_fn_552822

inputs!
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_550746w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿdd	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	
 
_user_specified_nameinputs

ý
D__inference_conv2d_6_layer_call_and_return_conditional_losses_552555

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
¼
ü
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550615
input_43
conv2d_transpose_2_550604:	'
conv2d_transpose_2_550606:	3
conv2d_transpose_3_550609:		'
conv2d_transpose_3_550611:	
identity¢*conv2d_transpose_2/StatefulPartitionedCall¢*conv2d_transpose_3/StatefulPartitionedCall¦
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_transpose_2_550604conv2d_transpose_2_550606*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_550467Ò
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_550609conv2d_transpose_3_550611*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_550512
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	 
NoOpNoOp+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input_4
Ì
¨
3__inference_conv2d_transpose_3_layer_call_fn_552693

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_550512
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
Ï
Þ
+__inference_proto_dec2_layer_call_fn_552200

inputs!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550537w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Ý
¿
F__inference_proto_enc2_layer_call_and_return_conditional_losses_549963

inputs)
conv2d_3_549940:	
conv2d_3_549942:)
conv2d_4_549957:
conv2d_4_549959:
identity¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCallý
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_549940conv2d_3_549942*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_549939 
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_549957conv2d_4_549959*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_549956
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
NoOpNoOp!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22	: : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs
 
â
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550753

inputs1
conv2d_transpose_550726:		%
conv2d_transpose_550728:	3
conv2d_transpose_1_550731:		'
conv2d_transpose_1_550733:	)
conv2d_2_550747:	
conv2d_2_550749:
identity¢ conv2d_2/StatefulPartitionedCall¢(conv2d_transpose/StatefulPartitionedCall¢*conv2d_transpose_1/StatefulPartitionedCall
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_550726conv2d_transpose_550728*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_550667Ð
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_550731conv2d_transpose_1_550733*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_550712ª
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_2_550747conv2d_2_550749*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_550746
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÁ
NoOpNoOp!^conv2d_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ22	: : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs
È
¦
1__inference_conv2d_transpose_layer_call_fn_552736

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_550667
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
Ý
¿
F__inference_proto_enc2_layer_call_and_return_conditional_losses_550023

inputs)
conv2d_3_550012:	
conv2d_3_550014:)
conv2d_4_550017:
conv2d_4_550019:
identity¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCallý
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_550012conv2d_3_550014*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_549939 
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_550017conv2d_4_550019*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_549956
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
NoOpNoOp!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22	: : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs
¹
û
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550537

inputs3
conv2d_transpose_2_550526:	'
conv2d_transpose_2_550528:	3
conv2d_transpose_3_550531:		'
conv2d_transpose_3_550533:	
identity¢*conv2d_transpose_2/StatefulPartitionedCall¢*conv2d_transpose_3/StatefulPartitionedCall¥
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_2_550526conv2d_transpose_2_550528*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_550467Ò
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_550531conv2d_transpose_3_550533*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_550512
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	 
NoOpNoOp+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Ò
ß
+__inference_proto_enc2_layer_call_fn_549974
input_3!
unknown:	
	unknown_0:#
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc2_layer_call_and_return_conditional_losses_549963w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
!
_user_specified_name	input_3
¶
á
F__inference_proto_enc3_layer_call_and_return_conditional_losses_552073

inputsA
'conv2d_5_conv2d_readvariableop_resource:6
(conv2d_5_biasadd_readvariableop_resource:A
'conv2d_6_conv2d_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:
identity¢conv2d_5/BiasAdd/ReadVariableOp¢conv2d_5/Conv2D/ReadVariableOp¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0«
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22h
conv2d_5/EluEluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¿
conv2d_6/Conv2DConv2Dconv2d_5/Elu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22h
conv2d_6/EluEluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22q
IdentityIdentityconv2d_6/Elu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ì
NoOpNoOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
à
À
F__inference_proto_enc2_layer_call_and_return_conditional_losses_550061
input_3)
conv2d_3_550050:	
conv2d_3_550052:)
conv2d_4_550055:
conv2d_4_550057:
identity¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCallþ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_3_550050conv2d_3_550052*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_549939 
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_550055conv2d_4_550057*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_549956
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
NoOpNoOp!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22	: : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
!
_user_specified_name	input_3

ý
D__inference_conv2d_5_layer_call_and_return_conditional_losses_552535

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
¹
û
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550577

inputs3
conv2d_transpose_2_550566:	'
conv2d_transpose_2_550568:	3
conv2d_transpose_3_550571:		'
conv2d_transpose_3_550573:	
identity¢*conv2d_transpose_2/StatefulPartitionedCall¢*conv2d_transpose_3/StatefulPartitionedCall¥
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_2_550566conv2d_transpose_2_550568*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_550467Ò
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_550571conv2d_transpose_3_550573*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_550512
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	 
NoOpNoOp+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Ê
º
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549907
input_1'
conv2d_549896:	
conv2d_549898:	)
conv2d_1_549901:		
conv2d_1_549903:	
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCallö
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_549896conv2d_549898*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_549785
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_549901conv2d_1_549903*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_549802
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿdd: : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
!
_user_specified_name	input_1
£
ã
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550886
input_21
conv2d_transpose_550870:		%
conv2d_transpose_550872:	3
conv2d_transpose_1_550875:		'
conv2d_transpose_1_550877:	)
conv2d_2_550880:	
conv2d_2_550882:
identity¢ conv2d_2/StatefulPartitionedCall¢(conv2d_transpose/StatefulPartitionedCall¢*conv2d_transpose_1/StatefulPartitionedCall
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallinput_2conv2d_transpose_550870conv2d_transpose_550872*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_550667Ð
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_550875conv2d_transpose_1_550877*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_550712ª
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_2_550880conv2d_2_550882*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_550746
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÁ
NoOpNoOp!^conv2d_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ22	: : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
!
_user_specified_name	input_2
Á!

N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_550312

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿh
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿz
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
®9
Ù
F__inference_proto_dec2_layer_call_and_return_conditional_losses_552257

inputsU
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource:	@
2conv2d_transpose_2_biasadd_readvariableop_resource:	U
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:		@
2conv2d_transpose_3_biasadd_readvariableop_resource:	
identity¢)conv2d_transpose_2/BiasAdd/ReadVariableOp¢2conv2d_transpose_2/conv2d_transpose/ReadVariableOp¢)conv2d_transpose_3/BiasAdd/ReadVariableOp¢2conv2d_transpose_3/conv2d_transpose/ReadVariableOpN
conv2d_transpose_2/ShapeShapeinputs*
T0*
_output_shapes
:p
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2\
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2\
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	è
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¶
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:	*
dtype0
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides

)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0À
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	|
conv2d_transpose_2/EluElu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	l
conv2d_transpose_3/ShapeShape$conv2d_transpose_2/Elu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2\
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	è
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¶
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0¡
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0$conv2d_transpose_2/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides

)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0À
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	|
conv2d_transpose_3/EluElu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	{
IdentityIdentity$conv2d_transpose_3/Elu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
NoOpNoOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
¹
û
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550377

inputs3
conv2d_transpose_4_550366:'
conv2d_transpose_4_550368:3
conv2d_transpose_5_550371:'
conv2d_transpose_5_550373:
identity¢*conv2d_transpose_4/StatefulPartitionedCall¢*conv2d_transpose_5/StatefulPartitionedCall¥
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_4_550366conv2d_transpose_4_550368*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_550267Ò
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_550371conv2d_transpose_5_550373*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_550312
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
NoOpNoOp+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Ý
¿
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550117

inputs)
conv2d_5_550094:
conv2d_5_550096:)
conv2d_6_550111:
conv2d_6_550113:
identity¢ conv2d_5/StatefulPartitionedCall¢ conv2d_6/StatefulPartitionedCallý
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_550094conv2d_5_550096*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_550093 
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_550111conv2d_6_550113*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_550110
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
NoOpNoOp!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Á!

N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_550712

inputsB
(conv2d_transpose_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :	y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
 
â
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550816

inputs1
conv2d_transpose_550800:		%
conv2d_transpose_550802:	3
conv2d_transpose_1_550805:		'
conv2d_transpose_1_550807:	)
conv2d_2_550810:	
conv2d_2_550812:
identity¢ conv2d_2/StatefulPartitionedCall¢(conv2d_transpose/StatefulPartitionedCall¢*conv2d_transpose_1/StatefulPartitionedCall
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_550800conv2d_transpose_550802*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *U
fPRN
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_550667Ð
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0conv2d_transpose_1_550805conv2d_transpose_1_550807*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_550712ª
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0conv2d_2_550810conv2d_2_550812*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_2_layer_call_and_return_conditional_losses_550746
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÁ
NoOpNoOp!^conv2d_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ22	: : : : : : 2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs

ý
D__inference_conv2d_4_layer_call_and_return_conditional_losses_552515

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs

ý
D__inference_conv2d_3_layer_call_and_return_conditional_losses_549939

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs
Ï
Þ
+__inference_proto_dec3_layer_call_fn_552099

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550377w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
¶
á
F__inference_proto_enc2_layer_call_and_return_conditional_losses_552011

inputsA
'conv2d_3_conv2d_readvariableop_resource:	6
(conv2d_3_biasadd_readvariableop_resource:A
'conv2d_4_conv2d_readvariableop_resource:6
(conv2d_4_biasadd_readvariableop_resource:
identity¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢conv2d_4/BiasAdd/ReadVariableOp¢conv2d_4/Conv2D/ReadVariableOp
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0«
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22h
conv2d_3/EluEluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¿
conv2d_4/Conv2DConv2Dconv2d_3/Elu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22h
conv2d_4/EluEluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22q
IdentityIdentityconv2d_4/Elu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ì
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22	: : : : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs
ë

0__inference_proto_ae_123321_layer_call_fn_551543

inputs!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:	
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:	

unknown_16:	$

unknown_17:		

unknown_18:	$

unknown_19:		

unknown_20:	$

unknown_21:		

unknown_22:	$

unknown_23:	

unknown_24:
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551128w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿdd: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
Ê¿
Ä!
!__inference__wrapped_model_549767
proto_enc1_inputZ
@proto_ae_123321_proto_enc1_conv2d_conv2d_readvariableop_resource:	O
Aproto_ae_123321_proto_enc1_conv2d_biasadd_readvariableop_resource:	\
Bproto_ae_123321_proto_enc1_conv2d_1_conv2d_readvariableop_resource:		Q
Cproto_ae_123321_proto_enc1_conv2d_1_biasadd_readvariableop_resource:	\
Bproto_ae_123321_proto_enc2_conv2d_3_conv2d_readvariableop_resource:	Q
Cproto_ae_123321_proto_enc2_conv2d_3_biasadd_readvariableop_resource:\
Bproto_ae_123321_proto_enc2_conv2d_4_conv2d_readvariableop_resource:Q
Cproto_ae_123321_proto_enc2_conv2d_4_biasadd_readvariableop_resource:\
Bproto_ae_123321_proto_enc3_conv2d_5_conv2d_readvariableop_resource:Q
Cproto_ae_123321_proto_enc3_conv2d_5_biasadd_readvariableop_resource:\
Bproto_ae_123321_proto_enc3_conv2d_6_conv2d_readvariableop_resource:Q
Cproto_ae_123321_proto_enc3_conv2d_6_biasadd_readvariableop_resource:p
Vproto_ae_123321_proto_dec3_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:[
Mproto_ae_123321_proto_dec3_conv2d_transpose_4_biasadd_readvariableop_resource:p
Vproto_ae_123321_proto_dec3_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:[
Mproto_ae_123321_proto_dec3_conv2d_transpose_5_biasadd_readvariableop_resource:p
Vproto_ae_123321_proto_dec2_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:	[
Mproto_ae_123321_proto_dec2_conv2d_transpose_2_biasadd_readvariableop_resource:	p
Vproto_ae_123321_proto_dec2_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:		[
Mproto_ae_123321_proto_dec2_conv2d_transpose_3_biasadd_readvariableop_resource:	n
Tproto_ae_123321_proto_dec1_conv2d_transpose_conv2d_transpose_readvariableop_resource:		Y
Kproto_ae_123321_proto_dec1_conv2d_transpose_biasadd_readvariableop_resource:	p
Vproto_ae_123321_proto_dec1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:		[
Mproto_ae_123321_proto_dec1_conv2d_transpose_1_biasadd_readvariableop_resource:	\
Bproto_ae_123321_proto_dec1_conv2d_2_conv2d_readvariableop_resource:	Q
Cproto_ae_123321_proto_dec1_conv2d_2_biasadd_readvariableop_resource:
identity¢:proto_ae_123321/proto_dec1/conv2d_2/BiasAdd/ReadVariableOp¢9proto_ae_123321/proto_dec1/conv2d_2/Conv2D/ReadVariableOp¢Bproto_ae_123321/proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOp¢Kproto_ae_123321/proto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOp¢Dproto_ae_123321/proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOp¢Mproto_ae_123321/proto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp¢Dproto_ae_123321/proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOp¢Mproto_ae_123321/proto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp¢Dproto_ae_123321/proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOp¢Mproto_ae_123321/proto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp¢Dproto_ae_123321/proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOp¢Mproto_ae_123321/proto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOp¢Dproto_ae_123321/proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOp¢Mproto_ae_123321/proto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOp¢8proto_ae_123321/proto_enc1/conv2d/BiasAdd/ReadVariableOp¢7proto_ae_123321/proto_enc1/conv2d/Conv2D/ReadVariableOp¢:proto_ae_123321/proto_enc1/conv2d_1/BiasAdd/ReadVariableOp¢9proto_ae_123321/proto_enc1/conv2d_1/Conv2D/ReadVariableOp¢:proto_ae_123321/proto_enc2/conv2d_3/BiasAdd/ReadVariableOp¢9proto_ae_123321/proto_enc2/conv2d_3/Conv2D/ReadVariableOp¢:proto_ae_123321/proto_enc2/conv2d_4/BiasAdd/ReadVariableOp¢9proto_ae_123321/proto_enc2/conv2d_4/Conv2D/ReadVariableOp¢:proto_ae_123321/proto_enc3/conv2d_5/BiasAdd/ReadVariableOp¢9proto_ae_123321/proto_enc3/conv2d_5/Conv2D/ReadVariableOp¢:proto_ae_123321/proto_enc3/conv2d_6/BiasAdd/ReadVariableOp¢9proto_ae_123321/proto_enc3/conv2d_6/Conv2D/ReadVariableOpÀ
7proto_ae_123321/proto_enc1/conv2d/Conv2D/ReadVariableOpReadVariableOp@proto_ae_123321_proto_enc1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0ç
(proto_ae_123321/proto_enc1/conv2d/Conv2DConv2Dproto_enc1_input?proto_ae_123321/proto_enc1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides
¶
8proto_ae_123321/proto_enc1/conv2d/BiasAdd/ReadVariableOpReadVariableOpAproto_ae_123321_proto_enc1_conv2d_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0ã
)proto_ae_123321/proto_enc1/conv2d/BiasAddBiasAdd1proto_ae_123321/proto_enc1/conv2d/Conv2D:output:0@proto_ae_123321/proto_enc1/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
%proto_ae_123321/proto_enc1/conv2d/EluElu2proto_ae_123321/proto_enc1/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	Ä
9proto_ae_123321/proto_enc1/conv2d_1/Conv2D/ReadVariableOpReadVariableOpBproto_ae_123321_proto_enc1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0
*proto_ae_123321/proto_enc1/conv2d_1/Conv2DConv2D3proto_ae_123321/proto_enc1/conv2d/Elu:activations:0Aproto_ae_123321/proto_enc1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides
º
:proto_ae_123321/proto_enc1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpCproto_ae_123321_proto_enc1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0é
+proto_ae_123321/proto_enc1/conv2d_1/BiasAddBiasAdd3proto_ae_123321/proto_enc1/conv2d_1/Conv2D:output:0Bproto_ae_123321/proto_enc1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
'proto_ae_123321/proto_enc1/conv2d_1/EluElu4proto_ae_123321/proto_enc1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	Ä
9proto_ae_123321/proto_enc2/conv2d_3/Conv2D/ReadVariableOpReadVariableOpBproto_ae_123321_proto_enc2_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0
*proto_ae_123321/proto_enc2/conv2d_3/Conv2DConv2D5proto_ae_123321/proto_enc1/conv2d_1/Elu:activations:0Aproto_ae_123321/proto_enc2/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
º
:proto_ae_123321/proto_enc2/conv2d_3/BiasAdd/ReadVariableOpReadVariableOpCproto_ae_123321_proto_enc2_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0é
+proto_ae_123321/proto_enc2/conv2d_3/BiasAddBiasAdd3proto_ae_123321/proto_enc2/conv2d_3/Conv2D:output:0Bproto_ae_123321/proto_enc2/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
'proto_ae_123321/proto_enc2/conv2d_3/EluElu4proto_ae_123321/proto_enc2/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ä
9proto_ae_123321/proto_enc2/conv2d_4/Conv2D/ReadVariableOpReadVariableOpBproto_ae_123321_proto_enc2_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
*proto_ae_123321/proto_enc2/conv2d_4/Conv2DConv2D5proto_ae_123321/proto_enc2/conv2d_3/Elu:activations:0Aproto_ae_123321/proto_enc2/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
º
:proto_ae_123321/proto_enc2/conv2d_4/BiasAdd/ReadVariableOpReadVariableOpCproto_ae_123321_proto_enc2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0é
+proto_ae_123321/proto_enc2/conv2d_4/BiasAddBiasAdd3proto_ae_123321/proto_enc2/conv2d_4/Conv2D:output:0Bproto_ae_123321/proto_enc2/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
'proto_ae_123321/proto_enc2/conv2d_4/EluElu4proto_ae_123321/proto_enc2/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ä
9proto_ae_123321/proto_enc3/conv2d_5/Conv2D/ReadVariableOpReadVariableOpBproto_ae_123321_proto_enc3_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
*proto_ae_123321/proto_enc3/conv2d_5/Conv2DConv2D5proto_ae_123321/proto_enc2/conv2d_4/Elu:activations:0Aproto_ae_123321/proto_enc3/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
º
:proto_ae_123321/proto_enc3/conv2d_5/BiasAdd/ReadVariableOpReadVariableOpCproto_ae_123321_proto_enc3_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0é
+proto_ae_123321/proto_enc3/conv2d_5/BiasAddBiasAdd3proto_ae_123321/proto_enc3/conv2d_5/Conv2D:output:0Bproto_ae_123321/proto_enc3/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
'proto_ae_123321/proto_enc3/conv2d_5/EluElu4proto_ae_123321/proto_enc3/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ä
9proto_ae_123321/proto_enc3/conv2d_6/Conv2D/ReadVariableOpReadVariableOpBproto_ae_123321_proto_enc3_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
*proto_ae_123321/proto_enc3/conv2d_6/Conv2DConv2D5proto_ae_123321/proto_enc3/conv2d_5/Elu:activations:0Aproto_ae_123321/proto_enc3/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
º
:proto_ae_123321/proto_enc3/conv2d_6/BiasAdd/ReadVariableOpReadVariableOpCproto_ae_123321_proto_enc3_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0é
+proto_ae_123321/proto_enc3/conv2d_6/BiasAddBiasAdd3proto_ae_123321/proto_enc3/conv2d_6/Conv2D:output:0Bproto_ae_123321/proto_enc3/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
'proto_ae_123321/proto_enc3/conv2d_6/EluElu4proto_ae_123321/proto_enc3/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
3proto_ae_123321/proto_dec3/conv2d_transpose_4/ShapeShape5proto_ae_123321/proto_enc3/conv2d_6/Elu:activations:0*
T0*
_output_shapes
:
Aproto_ae_123321/proto_dec3/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Cproto_ae_123321/proto_dec3/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Cproto_ae_123321/proto_dec3/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:·
;proto_ae_123321/proto_dec3/conv2d_transpose_4/strided_sliceStridedSlice<proto_ae_123321/proto_dec3/conv2d_transpose_4/Shape:output:0Jproto_ae_123321/proto_dec3/conv2d_transpose_4/strided_slice/stack:output:0Lproto_ae_123321/proto_dec3/conv2d_transpose_4/strided_slice/stack_1:output:0Lproto_ae_123321/proto_dec3/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5proto_ae_123321/proto_dec3/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2w
5proto_ae_123321/proto_dec3/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2w
5proto_ae_123321/proto_dec3/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :ï
3proto_ae_123321/proto_dec3/conv2d_transpose_4/stackPackDproto_ae_123321/proto_dec3/conv2d_transpose_4/strided_slice:output:0>proto_ae_123321/proto_dec3/conv2d_transpose_4/stack/1:output:0>proto_ae_123321/proto_dec3/conv2d_transpose_4/stack/2:output:0>proto_ae_123321/proto_dec3/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:
Cproto_ae_123321/proto_dec3/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Eproto_ae_123321/proto_dec3/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Eproto_ae_123321/proto_dec3/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
=proto_ae_123321/proto_dec3/conv2d_transpose_4/strided_slice_1StridedSlice<proto_ae_123321/proto_dec3/conv2d_transpose_4/stack:output:0Lproto_ae_123321/proto_dec3/conv2d_transpose_4/strided_slice_1/stack:output:0Nproto_ae_123321/proto_dec3/conv2d_transpose_4/strided_slice_1/stack_1:output:0Nproto_ae_123321/proto_dec3/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskì
Mproto_ae_123321/proto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpVproto_ae_123321_proto_dec3_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0
>proto_ae_123321/proto_dec3/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput<proto_ae_123321/proto_dec3/conv2d_transpose_4/stack:output:0Uproto_ae_123321/proto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:05proto_ae_123321/proto_enc3/conv2d_6/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
Î
Dproto_ae_123321/proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOpMproto_ae_123321_proto_dec3_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
5proto_ae_123321/proto_dec3/conv2d_transpose_4/BiasAddBiasAddGproto_ae_123321/proto_dec3/conv2d_transpose_4/conv2d_transpose:output:0Lproto_ae_123321/proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22²
1proto_ae_123321/proto_dec3/conv2d_transpose_4/EluElu>proto_ae_123321/proto_dec3/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¢
3proto_ae_123321/proto_dec3/conv2d_transpose_5/ShapeShape?proto_ae_123321/proto_dec3/conv2d_transpose_4/Elu:activations:0*
T0*
_output_shapes
:
Aproto_ae_123321/proto_dec3/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Cproto_ae_123321/proto_dec3/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Cproto_ae_123321/proto_dec3/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:·
;proto_ae_123321/proto_dec3/conv2d_transpose_5/strided_sliceStridedSlice<proto_ae_123321/proto_dec3/conv2d_transpose_5/Shape:output:0Jproto_ae_123321/proto_dec3/conv2d_transpose_5/strided_slice/stack:output:0Lproto_ae_123321/proto_dec3/conv2d_transpose_5/strided_slice/stack_1:output:0Lproto_ae_123321/proto_dec3/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5proto_ae_123321/proto_dec3/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2w
5proto_ae_123321/proto_dec3/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2w
5proto_ae_123321/proto_dec3/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :ï
3proto_ae_123321/proto_dec3/conv2d_transpose_5/stackPackDproto_ae_123321/proto_dec3/conv2d_transpose_5/strided_slice:output:0>proto_ae_123321/proto_dec3/conv2d_transpose_5/stack/1:output:0>proto_ae_123321/proto_dec3/conv2d_transpose_5/stack/2:output:0>proto_ae_123321/proto_dec3/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:
Cproto_ae_123321/proto_dec3/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Eproto_ae_123321/proto_dec3/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Eproto_ae_123321/proto_dec3/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
=proto_ae_123321/proto_dec3/conv2d_transpose_5/strided_slice_1StridedSlice<proto_ae_123321/proto_dec3/conv2d_transpose_5/stack:output:0Lproto_ae_123321/proto_dec3/conv2d_transpose_5/strided_slice_1/stack:output:0Nproto_ae_123321/proto_dec3/conv2d_transpose_5/strided_slice_1/stack_1:output:0Nproto_ae_123321/proto_dec3/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskì
Mproto_ae_123321/proto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpVproto_ae_123321_proto_dec3_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0
>proto_ae_123321/proto_dec3/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput<proto_ae_123321/proto_dec3/conv2d_transpose_5/stack:output:0Uproto_ae_123321/proto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0?proto_ae_123321/proto_dec3/conv2d_transpose_4/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
Î
Dproto_ae_123321/proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOpMproto_ae_123321_proto_dec3_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
5proto_ae_123321/proto_dec3/conv2d_transpose_5/BiasAddBiasAddGproto_ae_123321/proto_dec3/conv2d_transpose_5/conv2d_transpose:output:0Lproto_ae_123321/proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22²
1proto_ae_123321/proto_dec3/conv2d_transpose_5/EluElu>proto_ae_123321/proto_dec3/conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¢
3proto_ae_123321/proto_dec2/conv2d_transpose_2/ShapeShape?proto_ae_123321/proto_dec3/conv2d_transpose_5/Elu:activations:0*
T0*
_output_shapes
:
Aproto_ae_123321/proto_dec2/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Cproto_ae_123321/proto_dec2/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Cproto_ae_123321/proto_dec2/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:·
;proto_ae_123321/proto_dec2/conv2d_transpose_2/strided_sliceStridedSlice<proto_ae_123321/proto_dec2/conv2d_transpose_2/Shape:output:0Jproto_ae_123321/proto_dec2/conv2d_transpose_2/strided_slice/stack:output:0Lproto_ae_123321/proto_dec2/conv2d_transpose_2/strided_slice/stack_1:output:0Lproto_ae_123321/proto_dec2/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5proto_ae_123321/proto_dec2/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2w
5proto_ae_123321/proto_dec2/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2w
5proto_ae_123321/proto_dec2/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	ï
3proto_ae_123321/proto_dec2/conv2d_transpose_2/stackPackDproto_ae_123321/proto_dec2/conv2d_transpose_2/strided_slice:output:0>proto_ae_123321/proto_dec2/conv2d_transpose_2/stack/1:output:0>proto_ae_123321/proto_dec2/conv2d_transpose_2/stack/2:output:0>proto_ae_123321/proto_dec2/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:
Cproto_ae_123321/proto_dec2/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Eproto_ae_123321/proto_dec2/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Eproto_ae_123321/proto_dec2/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
=proto_ae_123321/proto_dec2/conv2d_transpose_2/strided_slice_1StridedSlice<proto_ae_123321/proto_dec2/conv2d_transpose_2/stack:output:0Lproto_ae_123321/proto_dec2/conv2d_transpose_2/strided_slice_1/stack:output:0Nproto_ae_123321/proto_dec2/conv2d_transpose_2/strided_slice_1/stack_1:output:0Nproto_ae_123321/proto_dec2/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskì
Mproto_ae_123321/proto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpVproto_ae_123321_proto_dec2_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:	*
dtype0
>proto_ae_123321/proto_dec2/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput<proto_ae_123321/proto_dec2/conv2d_transpose_2/stack:output:0Uproto_ae_123321/proto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0?proto_ae_123321/proto_dec3/conv2d_transpose_5/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides
Î
Dproto_ae_123321/proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpMproto_ae_123321_proto_dec2_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
5proto_ae_123321/proto_dec2/conv2d_transpose_2/BiasAddBiasAddGproto_ae_123321/proto_dec2/conv2d_transpose_2/conv2d_transpose:output:0Lproto_ae_123321/proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	²
1proto_ae_123321/proto_dec2/conv2d_transpose_2/EluElu>proto_ae_123321/proto_dec2/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	¢
3proto_ae_123321/proto_dec2/conv2d_transpose_3/ShapeShape?proto_ae_123321/proto_dec2/conv2d_transpose_2/Elu:activations:0*
T0*
_output_shapes
:
Aproto_ae_123321/proto_dec2/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Cproto_ae_123321/proto_dec2/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Cproto_ae_123321/proto_dec2/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:·
;proto_ae_123321/proto_dec2/conv2d_transpose_3/strided_sliceStridedSlice<proto_ae_123321/proto_dec2/conv2d_transpose_3/Shape:output:0Jproto_ae_123321/proto_dec2/conv2d_transpose_3/strided_slice/stack:output:0Lproto_ae_123321/proto_dec2/conv2d_transpose_3/strided_slice/stack_1:output:0Lproto_ae_123321/proto_dec2/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5proto_ae_123321/proto_dec2/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2w
5proto_ae_123321/proto_dec2/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2w
5proto_ae_123321/proto_dec2/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	ï
3proto_ae_123321/proto_dec2/conv2d_transpose_3/stackPackDproto_ae_123321/proto_dec2/conv2d_transpose_3/strided_slice:output:0>proto_ae_123321/proto_dec2/conv2d_transpose_3/stack/1:output:0>proto_ae_123321/proto_dec2/conv2d_transpose_3/stack/2:output:0>proto_ae_123321/proto_dec2/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:
Cproto_ae_123321/proto_dec2/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Eproto_ae_123321/proto_dec2/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Eproto_ae_123321/proto_dec2/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
=proto_ae_123321/proto_dec2/conv2d_transpose_3/strided_slice_1StridedSlice<proto_ae_123321/proto_dec2/conv2d_transpose_3/stack:output:0Lproto_ae_123321/proto_dec2/conv2d_transpose_3/strided_slice_1/stack:output:0Nproto_ae_123321/proto_dec2/conv2d_transpose_3/strided_slice_1/stack_1:output:0Nproto_ae_123321/proto_dec2/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskì
Mproto_ae_123321/proto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpVproto_ae_123321_proto_dec2_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0
>proto_ae_123321/proto_dec2/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput<proto_ae_123321/proto_dec2/conv2d_transpose_3/stack:output:0Uproto_ae_123321/proto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0?proto_ae_123321/proto_dec2/conv2d_transpose_2/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides
Î
Dproto_ae_123321/proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOpMproto_ae_123321_proto_dec2_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
5proto_ae_123321/proto_dec2/conv2d_transpose_3/BiasAddBiasAddGproto_ae_123321/proto_dec2/conv2d_transpose_3/conv2d_transpose:output:0Lproto_ae_123321/proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	²
1proto_ae_123321/proto_dec2/conv2d_transpose_3/EluElu>proto_ae_123321/proto_dec2/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	 
1proto_ae_123321/proto_dec1/conv2d_transpose/ShapeShape?proto_ae_123321/proto_dec2/conv2d_transpose_3/Elu:activations:0*
T0*
_output_shapes
:
?proto_ae_123321/proto_dec1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Aproto_ae_123321/proto_dec1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Aproto_ae_123321/proto_dec1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:­
9proto_ae_123321/proto_dec1/conv2d_transpose/strided_sliceStridedSlice:proto_ae_123321/proto_dec1/conv2d_transpose/Shape:output:0Hproto_ae_123321/proto_dec1/conv2d_transpose/strided_slice/stack:output:0Jproto_ae_123321/proto_dec1/conv2d_transpose/strided_slice/stack_1:output:0Jproto_ae_123321/proto_dec1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_masku
3proto_ae_123321/proto_dec1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2u
3proto_ae_123321/proto_dec1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2u
3proto_ae_123321/proto_dec1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	å
1proto_ae_123321/proto_dec1/conv2d_transpose/stackPackBproto_ae_123321/proto_dec1/conv2d_transpose/strided_slice:output:0<proto_ae_123321/proto_dec1/conv2d_transpose/stack/1:output:0<proto_ae_123321/proto_dec1/conv2d_transpose/stack/2:output:0<proto_ae_123321/proto_dec1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:
Aproto_ae_123321/proto_dec1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Cproto_ae_123321/proto_dec1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Cproto_ae_123321/proto_dec1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:µ
;proto_ae_123321/proto_dec1/conv2d_transpose/strided_slice_1StridedSlice:proto_ae_123321/proto_dec1/conv2d_transpose/stack:output:0Jproto_ae_123321/proto_dec1/conv2d_transpose/strided_slice_1/stack:output:0Lproto_ae_123321/proto_dec1/conv2d_transpose/strided_slice_1/stack_1:output:0Lproto_ae_123321/proto_dec1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskè
Kproto_ae_123321/proto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpTproto_ae_123321_proto_dec1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0
<proto_ae_123321/proto_dec1/conv2d_transpose/conv2d_transposeConv2DBackpropInput:proto_ae_123321/proto_dec1/conv2d_transpose/stack:output:0Sproto_ae_123321/proto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0?proto_ae_123321/proto_dec2/conv2d_transpose_3/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides
Ê
Bproto_ae_123321/proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpKproto_ae_123321_proto_dec1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
3proto_ae_123321/proto_dec1/conv2d_transpose/BiasAddBiasAddEproto_ae_123321/proto_dec1/conv2d_transpose/conv2d_transpose:output:0Jproto_ae_123321/proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	®
/proto_ae_123321/proto_dec1/conv2d_transpose/EluElu<proto_ae_123321/proto_dec1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	 
3proto_ae_123321/proto_dec1/conv2d_transpose_1/ShapeShape=proto_ae_123321/proto_dec1/conv2d_transpose/Elu:activations:0*
T0*
_output_shapes
:
Aproto_ae_123321/proto_dec1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Cproto_ae_123321/proto_dec1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Cproto_ae_123321/proto_dec1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:·
;proto_ae_123321/proto_dec1/conv2d_transpose_1/strided_sliceStridedSlice<proto_ae_123321/proto_dec1/conv2d_transpose_1/Shape:output:0Jproto_ae_123321/proto_dec1/conv2d_transpose_1/strided_slice/stack:output:0Lproto_ae_123321/proto_dec1/conv2d_transpose_1/strided_slice/stack_1:output:0Lproto_ae_123321/proto_dec1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskw
5proto_ae_123321/proto_dec1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :dw
5proto_ae_123321/proto_dec1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :dw
5proto_ae_123321/proto_dec1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	ï
3proto_ae_123321/proto_dec1/conv2d_transpose_1/stackPackDproto_ae_123321/proto_dec1/conv2d_transpose_1/strided_slice:output:0>proto_ae_123321/proto_dec1/conv2d_transpose_1/stack/1:output:0>proto_ae_123321/proto_dec1/conv2d_transpose_1/stack/2:output:0>proto_ae_123321/proto_dec1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:
Cproto_ae_123321/proto_dec1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
Eproto_ae_123321/proto_dec1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
Eproto_ae_123321/proto_dec1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¿
=proto_ae_123321/proto_dec1/conv2d_transpose_1/strided_slice_1StridedSlice<proto_ae_123321/proto_dec1/conv2d_transpose_1/stack:output:0Lproto_ae_123321/proto_dec1/conv2d_transpose_1/strided_slice_1/stack:output:0Nproto_ae_123321/proto_dec1/conv2d_transpose_1/strided_slice_1/stack_1:output:0Nproto_ae_123321/proto_dec1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskì
Mproto_ae_123321/proto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpVproto_ae_123321_proto_dec1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0
>proto_ae_123321/proto_dec1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput<proto_ae_123321/proto_dec1/conv2d_transpose_1/stack:output:0Uproto_ae_123321/proto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0=proto_ae_123321/proto_dec1/conv2d_transpose/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	*
paddingSAME*
strides
Î
Dproto_ae_123321/proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpMproto_ae_123321_proto_dec1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
5proto_ae_123321/proto_dec1/conv2d_transpose_1/BiasAddBiasAddGproto_ae_123321/proto_dec1/conv2d_transpose_1/conv2d_transpose:output:0Lproto_ae_123321/proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	²
1proto_ae_123321/proto_dec1/conv2d_transpose_1/EluElu>proto_ae_123321/proto_dec1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	Ä
9proto_ae_123321/proto_dec1/conv2d_2/Conv2D/ReadVariableOpReadVariableOpBproto_ae_123321_proto_dec1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0
*proto_ae_123321/proto_dec1/conv2d_2/Conv2DConv2D?proto_ae_123321/proto_dec1/conv2d_transpose_1/Elu:activations:0Aproto_ae_123321/proto_dec1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingVALID*
strides
º
:proto_ae_123321/proto_dec1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOpCproto_ae_123321_proto_dec1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0é
+proto_ae_123321/proto_dec1/conv2d_2/BiasAddBiasAdd3proto_ae_123321/proto_dec1/conv2d_2/Conv2D:output:0Bproto_ae_123321/proto_dec1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
IdentityIdentity4proto_ae_123321/proto_dec1/conv2d_2/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
NoOpNoOp;^proto_ae_123321/proto_dec1/conv2d_2/BiasAdd/ReadVariableOp:^proto_ae_123321/proto_dec1/conv2d_2/Conv2D/ReadVariableOpC^proto_ae_123321/proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOpL^proto_ae_123321/proto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOpE^proto_ae_123321/proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOpN^proto_ae_123321/proto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpE^proto_ae_123321/proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOpN^proto_ae_123321/proto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOpE^proto_ae_123321/proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOpN^proto_ae_123321/proto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOpE^proto_ae_123321/proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOpN^proto_ae_123321/proto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOpE^proto_ae_123321/proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOpN^proto_ae_123321/proto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOp9^proto_ae_123321/proto_enc1/conv2d/BiasAdd/ReadVariableOp8^proto_ae_123321/proto_enc1/conv2d/Conv2D/ReadVariableOp;^proto_ae_123321/proto_enc1/conv2d_1/BiasAdd/ReadVariableOp:^proto_ae_123321/proto_enc1/conv2d_1/Conv2D/ReadVariableOp;^proto_ae_123321/proto_enc2/conv2d_3/BiasAdd/ReadVariableOp:^proto_ae_123321/proto_enc2/conv2d_3/Conv2D/ReadVariableOp;^proto_ae_123321/proto_enc2/conv2d_4/BiasAdd/ReadVariableOp:^proto_ae_123321/proto_enc2/conv2d_4/Conv2D/ReadVariableOp;^proto_ae_123321/proto_enc3/conv2d_5/BiasAdd/ReadVariableOp:^proto_ae_123321/proto_enc3/conv2d_5/Conv2D/ReadVariableOp;^proto_ae_123321/proto_enc3/conv2d_6/BiasAdd/ReadVariableOp:^proto_ae_123321/proto_enc3/conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿdd: : : : : : : : : : : : : : : : : : : : : : : : : : 2x
:proto_ae_123321/proto_dec1/conv2d_2/BiasAdd/ReadVariableOp:proto_ae_123321/proto_dec1/conv2d_2/BiasAdd/ReadVariableOp2v
9proto_ae_123321/proto_dec1/conv2d_2/Conv2D/ReadVariableOp9proto_ae_123321/proto_dec1/conv2d_2/Conv2D/ReadVariableOp2
Bproto_ae_123321/proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOpBproto_ae_123321/proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOp2
Kproto_ae_123321/proto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOpKproto_ae_123321/proto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOp2
Dproto_ae_123321/proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOpDproto_ae_123321/proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOp2
Mproto_ae_123321/proto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpMproto_ae_123321/proto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2
Dproto_ae_123321/proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOpDproto_ae_123321/proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOp2
Mproto_ae_123321/proto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOpMproto_ae_123321/proto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2
Dproto_ae_123321/proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOpDproto_ae_123321/proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOp2
Mproto_ae_123321/proto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOpMproto_ae_123321/proto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2
Dproto_ae_123321/proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOpDproto_ae_123321/proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOp2
Mproto_ae_123321/proto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOpMproto_ae_123321/proto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2
Dproto_ae_123321/proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOpDproto_ae_123321/proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOp2
Mproto_ae_123321/proto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOpMproto_ae_123321/proto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2t
8proto_ae_123321/proto_enc1/conv2d/BiasAdd/ReadVariableOp8proto_ae_123321/proto_enc1/conv2d/BiasAdd/ReadVariableOp2r
7proto_ae_123321/proto_enc1/conv2d/Conv2D/ReadVariableOp7proto_ae_123321/proto_enc1/conv2d/Conv2D/ReadVariableOp2x
:proto_ae_123321/proto_enc1/conv2d_1/BiasAdd/ReadVariableOp:proto_ae_123321/proto_enc1/conv2d_1/BiasAdd/ReadVariableOp2v
9proto_ae_123321/proto_enc1/conv2d_1/Conv2D/ReadVariableOp9proto_ae_123321/proto_enc1/conv2d_1/Conv2D/ReadVariableOp2x
:proto_ae_123321/proto_enc2/conv2d_3/BiasAdd/ReadVariableOp:proto_ae_123321/proto_enc2/conv2d_3/BiasAdd/ReadVariableOp2v
9proto_ae_123321/proto_enc2/conv2d_3/Conv2D/ReadVariableOp9proto_ae_123321/proto_enc2/conv2d_3/Conv2D/ReadVariableOp2x
:proto_ae_123321/proto_enc2/conv2d_4/BiasAdd/ReadVariableOp:proto_ae_123321/proto_enc2/conv2d_4/BiasAdd/ReadVariableOp2v
9proto_ae_123321/proto_enc2/conv2d_4/Conv2D/ReadVariableOp9proto_ae_123321/proto_enc2/conv2d_4/Conv2D/ReadVariableOp2x
:proto_ae_123321/proto_enc3/conv2d_5/BiasAdd/ReadVariableOp:proto_ae_123321/proto_enc3/conv2d_5/BiasAdd/ReadVariableOp2v
9proto_ae_123321/proto_enc3/conv2d_5/Conv2D/ReadVariableOp9proto_ae_123321/proto_enc3/conv2d_5/Conv2D/ReadVariableOp2x
:proto_ae_123321/proto_enc3/conv2d_6/BiasAdd/ReadVariableOp:proto_ae_123321/proto_enc3/conv2d_6/BiasAdd/ReadVariableOp2v
9proto_ae_123321/proto_enc3/conv2d_6/Conv2D/ReadVariableOp9proto_ae_123321/proto_enc3/conv2d_6/Conv2D/ReadVariableOp:a ]
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
*
_user_specified_nameproto_enc1_input
³Ö
ê9
"__inference__traced_restore_553387
file_prefix8
assignvariableop_conv2d_kernel:	,
assignvariableop_1_conv2d_bias:	<
"assignvariableop_2_conv2d_1_kernel:		.
 assignvariableop_3_conv2d_1_bias:	<
"assignvariableop_4_conv2d_3_kernel:	.
 assignvariableop_5_conv2d_3_bias:<
"assignvariableop_6_conv2d_4_kernel:.
 assignvariableop_7_conv2d_4_bias:<
"assignvariableop_8_conv2d_5_kernel:.
 assignvariableop_9_conv2d_5_bias:=
#assignvariableop_10_conv2d_6_kernel:/
!assignvariableop_11_conv2d_6_bias:G
-assignvariableop_12_conv2d_transpose_4_kernel:9
+assignvariableop_13_conv2d_transpose_4_bias:G
-assignvariableop_14_conv2d_transpose_5_kernel:9
+assignvariableop_15_conv2d_transpose_5_bias:G
-assignvariableop_16_conv2d_transpose_2_kernel:	9
+assignvariableop_17_conv2d_transpose_2_bias:	G
-assignvariableop_18_conv2d_transpose_3_kernel:		9
+assignvariableop_19_conv2d_transpose_3_bias:	E
+assignvariableop_20_conv2d_transpose_kernel:		7
)assignvariableop_21_conv2d_transpose_bias:	G
-assignvariableop_22_conv2d_transpose_1_kernel:		9
+assignvariableop_23_conv2d_transpose_1_bias:	=
#assignvariableop_24_conv2d_2_kernel:	/
!assignvariableop_25_conv2d_2_bias:'
assignvariableop_26_adam_iter:	 )
assignvariableop_27_adam_beta_1: )
assignvariableop_28_adam_beta_2: (
assignvariableop_29_adam_decay: 0
&assignvariableop_30_adam_learning_rate: %
assignvariableop_31_total_1: %
assignvariableop_32_count_1: #
assignvariableop_33_total: #
assignvariableop_34_count: B
(assignvariableop_35_adam_conv2d_kernel_m:	4
&assignvariableop_36_adam_conv2d_bias_m:	D
*assignvariableop_37_adam_conv2d_1_kernel_m:		6
(assignvariableop_38_adam_conv2d_1_bias_m:	D
*assignvariableop_39_adam_conv2d_3_kernel_m:	6
(assignvariableop_40_adam_conv2d_3_bias_m:D
*assignvariableop_41_adam_conv2d_4_kernel_m:6
(assignvariableop_42_adam_conv2d_4_bias_m:D
*assignvariableop_43_adam_conv2d_5_kernel_m:6
(assignvariableop_44_adam_conv2d_5_bias_m:D
*assignvariableop_45_adam_conv2d_6_kernel_m:6
(assignvariableop_46_adam_conv2d_6_bias_m:N
4assignvariableop_47_adam_conv2d_transpose_4_kernel_m:@
2assignvariableop_48_adam_conv2d_transpose_4_bias_m:N
4assignvariableop_49_adam_conv2d_transpose_5_kernel_m:@
2assignvariableop_50_adam_conv2d_transpose_5_bias_m:N
4assignvariableop_51_adam_conv2d_transpose_2_kernel_m:	@
2assignvariableop_52_adam_conv2d_transpose_2_bias_m:	N
4assignvariableop_53_adam_conv2d_transpose_3_kernel_m:		@
2assignvariableop_54_adam_conv2d_transpose_3_bias_m:	L
2assignvariableop_55_adam_conv2d_transpose_kernel_m:		>
0assignvariableop_56_adam_conv2d_transpose_bias_m:	N
4assignvariableop_57_adam_conv2d_transpose_1_kernel_m:		@
2assignvariableop_58_adam_conv2d_transpose_1_bias_m:	D
*assignvariableop_59_adam_conv2d_2_kernel_m:	6
(assignvariableop_60_adam_conv2d_2_bias_m:B
(assignvariableop_61_adam_conv2d_kernel_v:	4
&assignvariableop_62_adam_conv2d_bias_v:	D
*assignvariableop_63_adam_conv2d_1_kernel_v:		6
(assignvariableop_64_adam_conv2d_1_bias_v:	D
*assignvariableop_65_adam_conv2d_3_kernel_v:	6
(assignvariableop_66_adam_conv2d_3_bias_v:D
*assignvariableop_67_adam_conv2d_4_kernel_v:6
(assignvariableop_68_adam_conv2d_4_bias_v:D
*assignvariableop_69_adam_conv2d_5_kernel_v:6
(assignvariableop_70_adam_conv2d_5_bias_v:D
*assignvariableop_71_adam_conv2d_6_kernel_v:6
(assignvariableop_72_adam_conv2d_6_bias_v:N
4assignvariableop_73_adam_conv2d_transpose_4_kernel_v:@
2assignvariableop_74_adam_conv2d_transpose_4_bias_v:N
4assignvariableop_75_adam_conv2d_transpose_5_kernel_v:@
2assignvariableop_76_adam_conv2d_transpose_5_bias_v:N
4assignvariableop_77_adam_conv2d_transpose_2_kernel_v:	@
2assignvariableop_78_adam_conv2d_transpose_2_bias_v:	N
4assignvariableop_79_adam_conv2d_transpose_3_kernel_v:		@
2assignvariableop_80_adam_conv2d_transpose_3_bias_v:	L
2assignvariableop_81_adam_conv2d_transpose_kernel_v:		>
0assignvariableop_82_adam_conv2d_transpose_bias_v:	N
4assignvariableop_83_adam_conv2d_transpose_1_kernel_v:		@
2assignvariableop_84_adam_conv2d_transpose_1_bias_v:	D
*assignvariableop_85_adam_conv2d_2_kernel_v:	6
(assignvariableop_86_adam_conv2d_2_bias_v:
identity_88¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_75¢AssignVariableOp_76¢AssignVariableOp_77¢AssignVariableOp_78¢AssignVariableOp_79¢AssignVariableOp_8¢AssignVariableOp_80¢AssignVariableOp_81¢AssignVariableOp_82¢AssignVariableOp_83¢AssignVariableOp_84¢AssignVariableOp_85¢AssignVariableOp_86¢AssignVariableOp_9Ú(
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*(
valueö'Bó'XB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH£
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*Å
value»B¸XB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Ù
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ö
_output_shapesã
à::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*f
dtypes\
Z2X	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_3_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_3_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_4_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_4_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp"assignvariableop_8_conv2d_5_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp assignvariableop_9_conv2d_5_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp#assignvariableop_10_conv2d_6_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp!assignvariableop_11_conv2d_6_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOp-assignvariableop_12_conv2d_transpose_4_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp+assignvariableop_13_conv2d_transpose_4_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp-assignvariableop_14_conv2d_transpose_5_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp+assignvariableop_15_conv2d_transpose_5_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp-assignvariableop_16_conv2d_transpose_2_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp+assignvariableop_17_conv2d_transpose_2_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp-assignvariableop_18_conv2d_transpose_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp+assignvariableop_19_conv2d_transpose_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp+assignvariableop_20_conv2d_transpose_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp)assignvariableop_21_conv2d_transpose_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp-assignvariableop_22_conv2d_transpose_1_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp+assignvariableop_23_conv2d_transpose_1_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp#assignvariableop_24_conv2d_2_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp!assignvariableop_25_conv2d_2_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_iterIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_beta_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_beta_2Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOpassignvariableop_29_adam_decayIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_learning_rateIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOpassignvariableop_33_totalIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOpassignvariableop_34_countIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp(assignvariableop_35_adam_conv2d_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp&assignvariableop_36_adam_conv2d_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp*assignvariableop_37_adam_conv2d_1_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp(assignvariableop_38_adam_conv2d_1_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_39AssignVariableOp*assignvariableop_39_adam_conv2d_3_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_40AssignVariableOp(assignvariableop_40_adam_conv2d_3_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp*assignvariableop_41_adam_conv2d_4_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp(assignvariableop_42_adam_conv2d_4_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp*assignvariableop_43_adam_conv2d_5_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp(assignvariableop_44_adam_conv2d_5_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_45AssignVariableOp*assignvariableop_45_adam_conv2d_6_kernel_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_46AssignVariableOp(assignvariableop_46_adam_conv2d_6_bias_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_47AssignVariableOp4assignvariableop_47_adam_conv2d_transpose_4_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_48AssignVariableOp2assignvariableop_48_adam_conv2d_transpose_4_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_49AssignVariableOp4assignvariableop_49_adam_conv2d_transpose_5_kernel_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_50AssignVariableOp2assignvariableop_50_adam_conv2d_transpose_5_bias_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_51AssignVariableOp4assignvariableop_51_adam_conv2d_transpose_2_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_52AssignVariableOp2assignvariableop_52_adam_conv2d_transpose_2_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_53AssignVariableOp4assignvariableop_53_adam_conv2d_transpose_3_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_54AssignVariableOp2assignvariableop_54_adam_conv2d_transpose_3_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_55AssignVariableOp2assignvariableop_55_adam_conv2d_transpose_kernel_mIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_56AssignVariableOp0assignvariableop_56_adam_conv2d_transpose_bias_mIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_57AssignVariableOp4assignvariableop_57_adam_conv2d_transpose_1_kernel_mIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_58AssignVariableOp2assignvariableop_58_adam_conv2d_transpose_1_bias_mIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_59AssignVariableOp*assignvariableop_59_adam_conv2d_2_kernel_mIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_60AssignVariableOp(assignvariableop_60_adam_conv2d_2_bias_mIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOp(assignvariableop_61_adam_conv2d_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOp&assignvariableop_62_adam_conv2d_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOp*assignvariableop_63_adam_conv2d_1_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOp(assignvariableop_64_adam_conv2d_1_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_65AssignVariableOp*assignvariableop_65_adam_conv2d_3_kernel_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_66AssignVariableOp(assignvariableop_66_adam_conv2d_3_bias_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_67AssignVariableOp*assignvariableop_67_adam_conv2d_4_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_68AssignVariableOp(assignvariableop_68_adam_conv2d_4_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_69AssignVariableOp*assignvariableop_69_adam_conv2d_5_kernel_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_70AssignVariableOp(assignvariableop_70_adam_conv2d_5_bias_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_conv2d_6_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_conv2d_6_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_73AssignVariableOp4assignvariableop_73_adam_conv2d_transpose_4_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_74AssignVariableOp2assignvariableop_74_adam_conv2d_transpose_4_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_75IdentityRestoreV2:tensors:75"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_75AssignVariableOp4assignvariableop_75_adam_conv2d_transpose_5_kernel_vIdentity_75:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_76IdentityRestoreV2:tensors:76"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_76AssignVariableOp2assignvariableop_76_adam_conv2d_transpose_5_bias_vIdentity_76:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_77IdentityRestoreV2:tensors:77"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_77AssignVariableOp4assignvariableop_77_adam_conv2d_transpose_2_kernel_vIdentity_77:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_78IdentityRestoreV2:tensors:78"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_78AssignVariableOp2assignvariableop_78_adam_conv2d_transpose_2_bias_vIdentity_78:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_79IdentityRestoreV2:tensors:79"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_79AssignVariableOp4assignvariableop_79_adam_conv2d_transpose_3_kernel_vIdentity_79:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_80IdentityRestoreV2:tensors:80"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_80AssignVariableOp2assignvariableop_80_adam_conv2d_transpose_3_bias_vIdentity_80:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_81IdentityRestoreV2:tensors:81"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_81AssignVariableOp2assignvariableop_81_adam_conv2d_transpose_kernel_vIdentity_81:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_82IdentityRestoreV2:tensors:82"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_82AssignVariableOp0assignvariableop_82_adam_conv2d_transpose_bias_vIdentity_82:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_83IdentityRestoreV2:tensors:83"/device:CPU:0*
T0*
_output_shapes
:¥
AssignVariableOp_83AssignVariableOp4assignvariableop_83_adam_conv2d_transpose_1_kernel_vIdentity_83:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_84IdentityRestoreV2:tensors:84"/device:CPU:0*
T0*
_output_shapes
:£
AssignVariableOp_84AssignVariableOp2assignvariableop_84_adam_conv2d_transpose_1_bias_vIdentity_84:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_85IdentityRestoreV2:tensors:85"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_85AssignVariableOp*assignvariableop_85_adam_conv2d_2_kernel_vIdentity_85:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_86IdentityRestoreV2:tensors:86"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_86AssignVariableOp(assignvariableop_86_adam_conv2d_2_bias_vIdentity_86:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 É
Identity_87Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_88IdentityIdentity_87:output:0^NoOp_1*
T0*
_output_shapes
: ¶
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_75^AssignVariableOp_76^AssignVariableOp_77^AssignVariableOp_78^AssignVariableOp_79^AssignVariableOp_8^AssignVariableOp_80^AssignVariableOp_81^AssignVariableOp_82^AssignVariableOp_83^AssignVariableOp_84^AssignVariableOp_85^AssignVariableOp_86^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_88Identity_88:output:0*Å
_input_shapes³
°: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742*
AssignVariableOp_75AssignVariableOp_752*
AssignVariableOp_76AssignVariableOp_762*
AssignVariableOp_77AssignVariableOp_772*
AssignVariableOp_78AssignVariableOp_782*
AssignVariableOp_79AssignVariableOp_792(
AssignVariableOp_8AssignVariableOp_82*
AssignVariableOp_80AssignVariableOp_802*
AssignVariableOp_81AssignVariableOp_812*
AssignVariableOp_82AssignVariableOp_822*
AssignVariableOp_83AssignVariableOp_832*
AssignVariableOp_84AssignVariableOp_842*
AssignVariableOp_85AssignVariableOp_852*
AssignVariableOp_86AssignVariableOp_862(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ç
¹
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549869

inputs'
conv2d_549858:	
conv2d_549860:	)
conv2d_1_549863:		
conv2d_1_549865:	
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCallõ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_549858conv2d_549860*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_549785
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_549863conv2d_1_549865*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_549802
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿdd: : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
®9
Ù
F__inference_proto_dec3_layer_call_and_return_conditional_losses_552143

inputsU
;conv2d_transpose_4_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_4_biasadd_readvariableop_resource:U
;conv2d_transpose_5_conv2d_transpose_readvariableop_resource:@
2conv2d_transpose_5_biasadd_readvariableop_resource:
identity¢)conv2d_transpose_4/BiasAdd/ReadVariableOp¢2conv2d_transpose_4/conv2d_transpose/ReadVariableOp¢)conv2d_transpose_5/BiasAdd/ReadVariableOp¢2conv2d_transpose_5/conv2d_transpose/ReadVariableOpN
conv2d_transpose_4/ShapeShapeinputs*
T0*
_output_shapes
:p
&conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_4/strided_sliceStridedSlice!conv2d_transpose_4/Shape:output:0/conv2d_transpose_4/strided_slice/stack:output:01conv2d_transpose_4/strided_slice/stack_1:output:01conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2\
conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2\
conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :è
conv2d_transpose_4/stackPack)conv2d_transpose_4/strided_slice:output:0#conv2d_transpose_4/stack/1:output:0#conv2d_transpose_4/stack/2:output:0#conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_4/strided_slice_1StridedSlice!conv2d_transpose_4/stack:output:01conv2d_transpose_4/strided_slice_1/stack:output:03conv2d_transpose_4/strided_slice_1/stack_1:output:03conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¶
2conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0
#conv2d_transpose_4/conv2d_transposeConv2DBackpropInput!conv2d_transpose_4/stack:output:0:conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0À
conv2d_transpose_4/BiasAddBiasAdd,conv2d_transpose_4/conv2d_transpose:output:01conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22|
conv2d_transpose_4/EluElu#conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22l
conv2d_transpose_5/ShapeShape$conv2d_transpose_4/Elu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_5/strided_sliceStridedSlice!conv2d_transpose_5/Shape:output:0/conv2d_transpose_5/strided_slice/stack:output:01conv2d_transpose_5/strided_slice/stack_1:output:01conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2\
conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2\
conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :è
conv2d_transpose_5/stackPack)conv2d_transpose_5/strided_slice:output:0#conv2d_transpose_5/stack/1:output:0#conv2d_transpose_5/stack/2:output:0#conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_5/strided_slice_1StridedSlice!conv2d_transpose_5/stack:output:01conv2d_transpose_5/strided_slice_1/stack:output:03conv2d_transpose_5/strided_slice_1/stack_1:output:03conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¶
2conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0¡
#conv2d_transpose_5/conv2d_transposeConv2DBackpropInput!conv2d_transpose_5/stack:output:0:conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0$conv2d_transpose_4/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

)conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0À
conv2d_transpose_5/BiasAddBiasAdd,conv2d_transpose_5/conv2d_transpose:output:01conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22|
conv2d_transpose_5/EluElu#conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22{
IdentityIdentity$conv2d_transpose_5/Elu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
NoOpNoOp*^conv2d_transpose_4/BiasAdd/ReadVariableOp3^conv2d_transpose_4/conv2d_transpose/ReadVariableOp*^conv2d_transpose_5/BiasAdd/ReadVariableOp3^conv2d_transpose_5/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2V
)conv2d_transpose_4/BiasAdd/ReadVariableOp)conv2d_transpose_4/BiasAdd/ReadVariableOp2h
2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2conv2d_transpose_4/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_5/BiasAdd/ReadVariableOp)conv2d_transpose_5/BiasAdd/ReadVariableOp2h
2conv2d_transpose_5/conv2d_transpose/ReadVariableOp2conv2d_transpose_5/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Ò
ß
+__inference_proto_dec2_layer_call_fn_550548
input_4!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550537w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input_4
ï

)__inference_conv2d_1_layer_call_fn_552464

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_549802w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs
à
À
F__inference_proto_enc2_layer_call_and_return_conditional_losses_550075
input_3)
conv2d_3_550064:	
conv2d_3_550066:)
conv2d_4_550069:
conv2d_4_550071:
identity¢ conv2d_3/StatefulPartitionedCall¢ conv2d_4/StatefulPartitionedCallþ
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinput_3conv2d_3_550064conv2d_3_550066*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_549939 
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_550069conv2d_4_550071*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_4_layer_call_and_return_conditional_losses_549956
IdentityIdentity)conv2d_4/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
NoOpNoOp!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22	: : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
!
_user_specified_name	input_3
Ò
ß
+__inference_proto_dec2_layer_call_fn_550601
input_4!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_4unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550577w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input_4
«	

+__inference_proto_dec1_layer_call_fn_552318

inputs!
unknown:		
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:	
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550753w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ22	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs
Ï
Þ
+__inference_proto_enc3_layer_call_fn_552037

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550177w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Ï
Þ
+__inference_proto_enc2_layer_call_fn_551962

inputs!
unknown:	
	unknown_0:#
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc2_layer_call_and_return_conditional_losses_549963w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs
Ì@

F__inference_proto_dec1_layer_call_and_return_conditional_losses_552435

inputsS
9conv2d_transpose_conv2d_transpose_readvariableop_resource:		>
0conv2d_transpose_biasadd_readvariableop_resource:	U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:		@
2conv2d_transpose_1_biasadd_readvariableop_resource:	A
'conv2d_2_conv2d_readvariableop_resource:	6
(conv2d_2_biasadd_readvariableop_resource:
identity¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢'conv2d_transpose/BiasAdd/ReadVariableOp¢0conv2d_transpose/conv2d_transpose/ReadVariableOp¢)conv2d_transpose_1/BiasAdd/ReadVariableOp¢2conv2d_transpose_1/conv2d_transpose/ReadVariableOpL
conv2d_transpose/ShapeShapeinputs*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	Þ
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask²
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0ý
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides

'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0º
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	x
conv2d_transpose/EluElu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	j
conv2d_transpose_1/ShapeShape"conv2d_transpose/Elu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	è
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¶
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0"conv2d_transpose/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	*
paddingSAME*
strides

)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0À
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	|
conv2d_transpose_1/EluElu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0Ê
conv2d_2/Conv2DConv2D$conv2d_transpose_1/Elu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿddp
IdentityIdentityconv2d_2/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÇ
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ22	: : : : : : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs
Á!

N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_552684

inputsB
(conv2d_transpose_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :	y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:	*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ì
¨
3__inference_conv2d_transpose_2_layer_call_fn_552650

inputs!
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_550467
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á!

N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_552598

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿh
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿz
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

¦
0__inference_proto_ae_123321_layer_call_fn_551240
proto_enc1_input!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:	
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:	

unknown_16:	$

unknown_17:		

unknown_18:	$

unknown_19:		

unknown_20:	$

unknown_21:		

unknown_22:	$

unknown_23:	

unknown_24:
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallproto_enc1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551128w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿdd: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
*
_user_specified_nameproto_enc1_input
à
À
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550215
input_5)
conv2d_5_550204:
conv2d_5_550206:)
conv2d_6_550209:
conv2d_6_550211:
identity¢ conv2d_5/StatefulPartitionedCall¢ conv2d_6/StatefulPartitionedCallþ
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinput_5conv2d_5_550204conv2d_5_550206*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_550093 
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_550209conv2d_6_550211*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_550110
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
NoOpNoOp!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input_5

ý
D__inference_conv2d_3_layer_call_and_return_conditional_losses_552495

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs
¶
á
F__inference_proto_enc3_layer_call_and_return_conditional_losses_552055

inputsA
'conv2d_5_conv2d_readvariableop_resource:6
(conv2d_5_biasadd_readvariableop_resource:A
'conv2d_6_conv2d_readvariableop_resource:6
(conv2d_6_biasadd_readvariableop_resource:
identity¢conv2d_5/BiasAdd/ReadVariableOp¢conv2d_5/Conv2D/ReadVariableOp¢conv2d_6/BiasAdd/ReadVariableOp¢conv2d_6/Conv2D/ReadVariableOp
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0«
conv2d_5/Conv2DConv2Dinputs&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22h
conv2d_5/EluEluconv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¿
conv2d_6/Conv2DConv2Dconv2d_5/Elu:activations:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22h
conv2d_6/EluEluconv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22q
IdentityIdentityconv2d_6/Elu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ì
NoOpNoOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
¾
¤
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551887

inputsJ
0proto_enc1_conv2d_conv2d_readvariableop_resource:	?
1proto_enc1_conv2d_biasadd_readvariableop_resource:	L
2proto_enc1_conv2d_1_conv2d_readvariableop_resource:		A
3proto_enc1_conv2d_1_biasadd_readvariableop_resource:	L
2proto_enc2_conv2d_3_conv2d_readvariableop_resource:	A
3proto_enc2_conv2d_3_biasadd_readvariableop_resource:L
2proto_enc2_conv2d_4_conv2d_readvariableop_resource:A
3proto_enc2_conv2d_4_biasadd_readvariableop_resource:L
2proto_enc3_conv2d_5_conv2d_readvariableop_resource:A
3proto_enc3_conv2d_5_biasadd_readvariableop_resource:L
2proto_enc3_conv2d_6_conv2d_readvariableop_resource:A
3proto_enc3_conv2d_6_biasadd_readvariableop_resource:`
Fproto_dec3_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:K
=proto_dec3_conv2d_transpose_4_biasadd_readvariableop_resource:`
Fproto_dec3_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:K
=proto_dec3_conv2d_transpose_5_biasadd_readvariableop_resource:`
Fproto_dec2_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:	K
=proto_dec2_conv2d_transpose_2_biasadd_readvariableop_resource:	`
Fproto_dec2_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:		K
=proto_dec2_conv2d_transpose_3_biasadd_readvariableop_resource:	^
Dproto_dec1_conv2d_transpose_conv2d_transpose_readvariableop_resource:		I
;proto_dec1_conv2d_transpose_biasadd_readvariableop_resource:	`
Fproto_dec1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:		K
=proto_dec1_conv2d_transpose_1_biasadd_readvariableop_resource:	L
2proto_dec1_conv2d_2_conv2d_readvariableop_resource:	A
3proto_dec1_conv2d_2_biasadd_readvariableop_resource:
identity¢*proto_dec1/conv2d_2/BiasAdd/ReadVariableOp¢)proto_dec1/conv2d_2/Conv2D/ReadVariableOp¢2proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOp¢;proto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOp¢4proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOp¢=proto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp¢4proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOp¢=proto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp¢4proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOp¢=proto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp¢4proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOp¢=proto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOp¢4proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOp¢=proto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOp¢(proto_enc1/conv2d/BiasAdd/ReadVariableOp¢'proto_enc1/conv2d/Conv2D/ReadVariableOp¢*proto_enc1/conv2d_1/BiasAdd/ReadVariableOp¢)proto_enc1/conv2d_1/Conv2D/ReadVariableOp¢*proto_enc2/conv2d_3/BiasAdd/ReadVariableOp¢)proto_enc2/conv2d_3/Conv2D/ReadVariableOp¢*proto_enc2/conv2d_4/BiasAdd/ReadVariableOp¢)proto_enc2/conv2d_4/Conv2D/ReadVariableOp¢*proto_enc3/conv2d_5/BiasAdd/ReadVariableOp¢)proto_enc3/conv2d_5/Conv2D/ReadVariableOp¢*proto_enc3/conv2d_6/BiasAdd/ReadVariableOp¢)proto_enc3/conv2d_6/Conv2D/ReadVariableOp 
'proto_enc1/conv2d/Conv2D/ReadVariableOpReadVariableOp0proto_enc1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0½
proto_enc1/conv2d/Conv2DConv2Dinputs/proto_enc1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides

(proto_enc1/conv2d/BiasAdd/ReadVariableOpReadVariableOp1proto_enc1_conv2d_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0³
proto_enc1/conv2d/BiasAddBiasAdd!proto_enc1/conv2d/Conv2D:output:00proto_enc1/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	z
proto_enc1/conv2d/EluElu"proto_enc1/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	¤
)proto_enc1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2proto_enc1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0Þ
proto_enc1/conv2d_1/Conv2DConv2D#proto_enc1/conv2d/Elu:activations:01proto_enc1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides

*proto_enc1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3proto_enc1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0¹
proto_enc1/conv2d_1/BiasAddBiasAdd#proto_enc1/conv2d_1/Conv2D:output:02proto_enc1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	~
proto_enc1/conv2d_1/EluElu$proto_enc1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	¤
)proto_enc2/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2proto_enc2_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0à
proto_enc2/conv2d_3/Conv2DConv2D%proto_enc1/conv2d_1/Elu:activations:01proto_enc2/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

*proto_enc2/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp3proto_enc2_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¹
proto_enc2/conv2d_3/BiasAddBiasAdd#proto_enc2/conv2d_3/Conv2D:output:02proto_enc2/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22~
proto_enc2/conv2d_3/EluElu$proto_enc2/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
)proto_enc2/conv2d_4/Conv2D/ReadVariableOpReadVariableOp2proto_enc2_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0à
proto_enc2/conv2d_4/Conv2DConv2D%proto_enc2/conv2d_3/Elu:activations:01proto_enc2/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

*proto_enc2/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp3proto_enc2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¹
proto_enc2/conv2d_4/BiasAddBiasAdd#proto_enc2/conv2d_4/Conv2D:output:02proto_enc2/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22~
proto_enc2/conv2d_4/EluElu$proto_enc2/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
)proto_enc3/conv2d_5/Conv2D/ReadVariableOpReadVariableOp2proto_enc3_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0à
proto_enc3/conv2d_5/Conv2DConv2D%proto_enc2/conv2d_4/Elu:activations:01proto_enc3/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

*proto_enc3/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp3proto_enc3_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¹
proto_enc3/conv2d_5/BiasAddBiasAdd#proto_enc3/conv2d_5/Conv2D:output:02proto_enc3/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22~
proto_enc3/conv2d_5/EluElu$proto_enc3/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
)proto_enc3/conv2d_6/Conv2D/ReadVariableOpReadVariableOp2proto_enc3_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0à
proto_enc3/conv2d_6/Conv2DConv2D%proto_enc3/conv2d_5/Elu:activations:01proto_enc3/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

*proto_enc3/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp3proto_enc3_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¹
proto_enc3/conv2d_6/BiasAddBiasAdd#proto_enc3/conv2d_6/Conv2D:output:02proto_enc3/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22~
proto_enc3/conv2d_6/EluElu$proto_enc3/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22x
#proto_dec3/conv2d_transpose_4/ShapeShape%proto_enc3/conv2d_6/Elu:activations:0*
T0*
_output_shapes
:{
1proto_dec3/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3proto_dec3/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3proto_dec3/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+proto_dec3/conv2d_transpose_4/strided_sliceStridedSlice,proto_dec3/conv2d_transpose_4/Shape:output:0:proto_dec3/conv2d_transpose_4/strided_slice/stack:output:0<proto_dec3/conv2d_transpose_4/strided_slice/stack_1:output:0<proto_dec3/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%proto_dec3/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2g
%proto_dec3/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2g
%proto_dec3/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
#proto_dec3/conv2d_transpose_4/stackPack4proto_dec3/conv2d_transpose_4/strided_slice:output:0.proto_dec3/conv2d_transpose_4/stack/1:output:0.proto_dec3/conv2d_transpose_4/stack/2:output:0.proto_dec3/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:}
3proto_dec3/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5proto_dec3/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5proto_dec3/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
-proto_dec3/conv2d_transpose_4/strided_slice_1StridedSlice,proto_dec3/conv2d_transpose_4/stack:output:0<proto_dec3/conv2d_transpose_4/strided_slice_1/stack:output:0>proto_dec3/conv2d_transpose_4/strided_slice_1/stack_1:output:0>proto_dec3/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÌ
=proto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpFproto_dec3_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0Ã
.proto_dec3/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput,proto_dec3/conv2d_transpose_4/stack:output:0Eproto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0%proto_enc3/conv2d_6/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
®
4proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp=proto_dec3_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0á
%proto_dec3/conv2d_transpose_4/BiasAddBiasAdd7proto_dec3/conv2d_transpose_4/conv2d_transpose:output:0<proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!proto_dec3/conv2d_transpose_4/EluElu.proto_dec3/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
#proto_dec3/conv2d_transpose_5/ShapeShape/proto_dec3/conv2d_transpose_4/Elu:activations:0*
T0*
_output_shapes
:{
1proto_dec3/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3proto_dec3/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3proto_dec3/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+proto_dec3/conv2d_transpose_5/strided_sliceStridedSlice,proto_dec3/conv2d_transpose_5/Shape:output:0:proto_dec3/conv2d_transpose_5/strided_slice/stack:output:0<proto_dec3/conv2d_transpose_5/strided_slice/stack_1:output:0<proto_dec3/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%proto_dec3/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2g
%proto_dec3/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2g
%proto_dec3/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
#proto_dec3/conv2d_transpose_5/stackPack4proto_dec3/conv2d_transpose_5/strided_slice:output:0.proto_dec3/conv2d_transpose_5/stack/1:output:0.proto_dec3/conv2d_transpose_5/stack/2:output:0.proto_dec3/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:}
3proto_dec3/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5proto_dec3/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5proto_dec3/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
-proto_dec3/conv2d_transpose_5/strided_slice_1StridedSlice,proto_dec3/conv2d_transpose_5/stack:output:0<proto_dec3/conv2d_transpose_5/strided_slice_1/stack:output:0>proto_dec3/conv2d_transpose_5/strided_slice_1/stack_1:output:0>proto_dec3/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÌ
=proto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpFproto_dec3_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0Í
.proto_dec3/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput,proto_dec3/conv2d_transpose_5/stack:output:0Eproto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0/proto_dec3/conv2d_transpose_4/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
®
4proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp=proto_dec3_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0á
%proto_dec3/conv2d_transpose_5/BiasAddBiasAdd7proto_dec3/conv2d_transpose_5/conv2d_transpose:output:0<proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!proto_dec3/conv2d_transpose_5/EluElu.proto_dec3/conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
#proto_dec2/conv2d_transpose_2/ShapeShape/proto_dec3/conv2d_transpose_5/Elu:activations:0*
T0*
_output_shapes
:{
1proto_dec2/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3proto_dec2/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3proto_dec2/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+proto_dec2/conv2d_transpose_2/strided_sliceStridedSlice,proto_dec2/conv2d_transpose_2/Shape:output:0:proto_dec2/conv2d_transpose_2/strided_slice/stack:output:0<proto_dec2/conv2d_transpose_2/strided_slice/stack_1:output:0<proto_dec2/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%proto_dec2/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2g
%proto_dec2/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2g
%proto_dec2/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	
#proto_dec2/conv2d_transpose_2/stackPack4proto_dec2/conv2d_transpose_2/strided_slice:output:0.proto_dec2/conv2d_transpose_2/stack/1:output:0.proto_dec2/conv2d_transpose_2/stack/2:output:0.proto_dec2/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:}
3proto_dec2/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5proto_dec2/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5proto_dec2/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
-proto_dec2/conv2d_transpose_2/strided_slice_1StridedSlice,proto_dec2/conv2d_transpose_2/stack:output:0<proto_dec2/conv2d_transpose_2/strided_slice_1/stack:output:0>proto_dec2/conv2d_transpose_2/strided_slice_1/stack_1:output:0>proto_dec2/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÌ
=proto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpFproto_dec2_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:	*
dtype0Í
.proto_dec2/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput,proto_dec2/conv2d_transpose_2/stack:output:0Eproto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0/proto_dec3/conv2d_transpose_5/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides
®
4proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp=proto_dec2_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0á
%proto_dec2/conv2d_transpose_2/BiasAddBiasAdd7proto_dec2/conv2d_transpose_2/conv2d_transpose:output:0<proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
!proto_dec2/conv2d_transpose_2/EluElu.proto_dec2/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
#proto_dec2/conv2d_transpose_3/ShapeShape/proto_dec2/conv2d_transpose_2/Elu:activations:0*
T0*
_output_shapes
:{
1proto_dec2/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3proto_dec2/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3proto_dec2/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+proto_dec2/conv2d_transpose_3/strided_sliceStridedSlice,proto_dec2/conv2d_transpose_3/Shape:output:0:proto_dec2/conv2d_transpose_3/strided_slice/stack:output:0<proto_dec2/conv2d_transpose_3/strided_slice/stack_1:output:0<proto_dec2/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%proto_dec2/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2g
%proto_dec2/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2g
%proto_dec2/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	
#proto_dec2/conv2d_transpose_3/stackPack4proto_dec2/conv2d_transpose_3/strided_slice:output:0.proto_dec2/conv2d_transpose_3/stack/1:output:0.proto_dec2/conv2d_transpose_3/stack/2:output:0.proto_dec2/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:}
3proto_dec2/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5proto_dec2/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5proto_dec2/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
-proto_dec2/conv2d_transpose_3/strided_slice_1StridedSlice,proto_dec2/conv2d_transpose_3/stack:output:0<proto_dec2/conv2d_transpose_3/strided_slice_1/stack:output:0>proto_dec2/conv2d_transpose_3/strided_slice_1/stack_1:output:0>proto_dec2/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÌ
=proto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpFproto_dec2_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0Í
.proto_dec2/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput,proto_dec2/conv2d_transpose_3/stack:output:0Eproto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0/proto_dec2/conv2d_transpose_2/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides
®
4proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp=proto_dec2_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0á
%proto_dec2/conv2d_transpose_3/BiasAddBiasAdd7proto_dec2/conv2d_transpose_3/conv2d_transpose:output:0<proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
!proto_dec2/conv2d_transpose_3/EluElu.proto_dec2/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
!proto_dec1/conv2d_transpose/ShapeShape/proto_dec2/conv2d_transpose_3/Elu:activations:0*
T0*
_output_shapes
:y
/proto_dec1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1proto_dec1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1proto_dec1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)proto_dec1/conv2d_transpose/strided_sliceStridedSlice*proto_dec1/conv2d_transpose/Shape:output:08proto_dec1/conv2d_transpose/strided_slice/stack:output:0:proto_dec1/conv2d_transpose/strided_slice/stack_1:output:0:proto_dec1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#proto_dec1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2e
#proto_dec1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2e
#proto_dec1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	
!proto_dec1/conv2d_transpose/stackPack2proto_dec1/conv2d_transpose/strided_slice:output:0,proto_dec1/conv2d_transpose/stack/1:output:0,proto_dec1/conv2d_transpose/stack/2:output:0,proto_dec1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:{
1proto_dec1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3proto_dec1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3proto_dec1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+proto_dec1/conv2d_transpose/strided_slice_1StridedSlice*proto_dec1/conv2d_transpose/stack:output:0:proto_dec1/conv2d_transpose/strided_slice_1/stack:output:0<proto_dec1/conv2d_transpose/strided_slice_1/stack_1:output:0<proto_dec1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÈ
;proto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpDproto_dec1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0Ç
,proto_dec1/conv2d_transpose/conv2d_transposeConv2DBackpropInput*proto_dec1/conv2d_transpose/stack:output:0Cproto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0/proto_dec2/conv2d_transpose_3/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides
ª
2proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp;proto_dec1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0Û
#proto_dec1/conv2d_transpose/BiasAddBiasAdd5proto_dec1/conv2d_transpose/conv2d_transpose:output:0:proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
proto_dec1/conv2d_transpose/EluElu,proto_dec1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
#proto_dec1/conv2d_transpose_1/ShapeShape-proto_dec1/conv2d_transpose/Elu:activations:0*
T0*
_output_shapes
:{
1proto_dec1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3proto_dec1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3proto_dec1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+proto_dec1/conv2d_transpose_1/strided_sliceStridedSlice,proto_dec1/conv2d_transpose_1/Shape:output:0:proto_dec1/conv2d_transpose_1/strided_slice/stack:output:0<proto_dec1/conv2d_transpose_1/strided_slice/stack_1:output:0<proto_dec1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%proto_dec1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :dg
%proto_dec1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :dg
%proto_dec1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	
#proto_dec1/conv2d_transpose_1/stackPack4proto_dec1/conv2d_transpose_1/strided_slice:output:0.proto_dec1/conv2d_transpose_1/stack/1:output:0.proto_dec1/conv2d_transpose_1/stack/2:output:0.proto_dec1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:}
3proto_dec1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5proto_dec1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5proto_dec1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
-proto_dec1/conv2d_transpose_1/strided_slice_1StridedSlice,proto_dec1/conv2d_transpose_1/stack:output:0<proto_dec1/conv2d_transpose_1/strided_slice_1/stack:output:0>proto_dec1/conv2d_transpose_1/strided_slice_1/stack_1:output:0>proto_dec1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÌ
=proto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpFproto_dec1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0Ë
.proto_dec1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput,proto_dec1/conv2d_transpose_1/stack:output:0Eproto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0-proto_dec1/conv2d_transpose/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	*
paddingSAME*
strides
®
4proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp=proto_dec1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0á
%proto_dec1/conv2d_transpose_1/BiasAddBiasAdd7proto_dec1/conv2d_transpose_1/conv2d_transpose:output:0<proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	
!proto_dec1/conv2d_transpose_1/EluElu.proto_dec1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	¤
)proto_dec1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2proto_dec1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0ë
proto_dec1/conv2d_2/Conv2DConv2D/proto_dec1/conv2d_transpose_1/Elu:activations:01proto_dec1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingVALID*
strides

*proto_dec1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3proto_dec1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¹
proto_dec1/conv2d_2/BiasAddBiasAdd#proto_dec1/conv2d_2/Conv2D:output:02proto_dec1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd{
IdentityIdentity$proto_dec1/conv2d_2/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd÷

NoOpNoOp+^proto_dec1/conv2d_2/BiasAdd/ReadVariableOp*^proto_dec1/conv2d_2/Conv2D/ReadVariableOp3^proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOp<^proto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOp5^proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOp>^proto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp5^proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOp>^proto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp5^proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOp>^proto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp5^proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOp>^proto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOp5^proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOp>^proto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOp)^proto_enc1/conv2d/BiasAdd/ReadVariableOp(^proto_enc1/conv2d/Conv2D/ReadVariableOp+^proto_enc1/conv2d_1/BiasAdd/ReadVariableOp*^proto_enc1/conv2d_1/Conv2D/ReadVariableOp+^proto_enc2/conv2d_3/BiasAdd/ReadVariableOp*^proto_enc2/conv2d_3/Conv2D/ReadVariableOp+^proto_enc2/conv2d_4/BiasAdd/ReadVariableOp*^proto_enc2/conv2d_4/Conv2D/ReadVariableOp+^proto_enc3/conv2d_5/BiasAdd/ReadVariableOp*^proto_enc3/conv2d_5/Conv2D/ReadVariableOp+^proto_enc3/conv2d_6/BiasAdd/ReadVariableOp*^proto_enc3/conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿdd: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*proto_dec1/conv2d_2/BiasAdd/ReadVariableOp*proto_dec1/conv2d_2/BiasAdd/ReadVariableOp2V
)proto_dec1/conv2d_2/Conv2D/ReadVariableOp)proto_dec1/conv2d_2/Conv2D/ReadVariableOp2h
2proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOp2proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOp2z
;proto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOp;proto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOp2l
4proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOp4proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOp2~
=proto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp=proto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2l
4proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOp4proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOp2~
=proto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp=proto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2l
4proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOp4proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOp2~
=proto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp=proto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2l
4proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOp4proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOp2~
=proto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOp=proto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2l
4proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOp4proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOp2~
=proto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOp=proto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2T
(proto_enc1/conv2d/BiasAdd/ReadVariableOp(proto_enc1/conv2d/BiasAdd/ReadVariableOp2R
'proto_enc1/conv2d/Conv2D/ReadVariableOp'proto_enc1/conv2d/Conv2D/ReadVariableOp2X
*proto_enc1/conv2d_1/BiasAdd/ReadVariableOp*proto_enc1/conv2d_1/BiasAdd/ReadVariableOp2V
)proto_enc1/conv2d_1/Conv2D/ReadVariableOp)proto_enc1/conv2d_1/Conv2D/ReadVariableOp2X
*proto_enc2/conv2d_3/BiasAdd/ReadVariableOp*proto_enc2/conv2d_3/BiasAdd/ReadVariableOp2V
)proto_enc2/conv2d_3/Conv2D/ReadVariableOp)proto_enc2/conv2d_3/Conv2D/ReadVariableOp2X
*proto_enc2/conv2d_4/BiasAdd/ReadVariableOp*proto_enc2/conv2d_4/BiasAdd/ReadVariableOp2V
)proto_enc2/conv2d_4/Conv2D/ReadVariableOp)proto_enc2/conv2d_4/Conv2D/ReadVariableOp2X
*proto_enc3/conv2d_5/BiasAdd/ReadVariableOp*proto_enc3/conv2d_5/BiasAdd/ReadVariableOp2V
)proto_enc3/conv2d_5/Conv2D/ReadVariableOp)proto_enc3/conv2d_5/Conv2D/ReadVariableOp2X
*proto_enc3/conv2d_6/BiasAdd/ReadVariableOp*proto_enc3/conv2d_6/BiasAdd/ReadVariableOp2V
)proto_enc3/conv2d_6/Conv2D/ReadVariableOp)proto_enc3/conv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
Ï
Þ
+__inference_proto_enc1_layer_call_fn_551913

inputs!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549869w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿdd: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
Ò
ß
+__inference_proto_dec3_layer_call_fn_550401
input_6!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550377w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input_6
Ó

$__inference_signature_wrapper_551429
proto_enc1_input!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:	
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:	

unknown_16:	$

unknown_17:		

unknown_18:	$

unknown_19:		

unknown_20:	$

unknown_21:		

unknown_22:	$

unknown_23:	

unknown_24:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallproto_enc1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 **
f%R#
!__inference__wrapped_model_549767w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿdd: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
*
_user_specified_nameproto_enc1_input
¨

ý
D__inference_conv2d_2_layer_call_and_return_conditional_losses_550746

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿddg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿddw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿdd	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	
 
_user_specified_nameinputs
þ

û
B__inference_conv2d_layer_call_and_return_conditional_losses_552455

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿdd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
¹
û
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550337

inputs3
conv2d_transpose_4_550326:'
conv2d_transpose_4_550328:3
conv2d_transpose_5_550331:'
conv2d_transpose_5_550333:
identity¢*conv2d_transpose_4/StatefulPartitionedCall¢*conv2d_transpose_5/StatefulPartitionedCall¥
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_transpose_4_550326conv2d_transpose_4_550328*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_550267Ò
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_550331conv2d_transpose_5_550333*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_550312
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
NoOpNoOp+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
¶
á
F__inference_proto_enc2_layer_call_and_return_conditional_losses_551993

inputsA
'conv2d_3_conv2d_readvariableop_resource:	6
(conv2d_3_biasadd_readvariableop_resource:A
'conv2d_4_conv2d_readvariableop_resource:6
(conv2d_4_biasadd_readvariableop_resource:
identity¢conv2d_3/BiasAdd/ReadVariableOp¢conv2d_3/Conv2D/ReadVariableOp¢conv2d_4/BiasAdd/ReadVariableOp¢conv2d_4/Conv2D/ReadVariableOp
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0«
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22h
conv2d_3/EluEluconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0¿
conv2d_4/Conv2DConv2Dconv2d_3/Elu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22h
conv2d_4/EluEluconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22q
IdentityIdentityconv2d_4/Elu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22Ì
NoOpNoOp ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22	: : : : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs
Ï
Þ
+__inference_proto_enc3_layer_call_fn_552024

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550117w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
ï

)__inference_conv2d_3_layer_call_fn_552484

inputs!
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_3_layer_call_and_return_conditional_losses_549939w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22	: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs
¨

ý
D__inference_conv2d_2_layer_call_and_return_conditional_losses_552832

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿddg
IdentityIdentityBiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿddw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿdd	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	
 
_user_specified_nameinputs
¼
ü
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550429
input_63
conv2d_transpose_4_550418:'
conv2d_transpose_4_550420:3
conv2d_transpose_5_550423:'
conv2d_transpose_5_550425:
identity¢*conv2d_transpose_4/StatefulPartitionedCall¢*conv2d_transpose_5/StatefulPartitionedCall¦
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCallinput_6conv2d_transpose_4_550418conv2d_transpose_4_550420*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_550267Ò
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_550423conv2d_transpose_5_550425*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_550312
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
NoOpNoOp+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input_6
Ü)
È

K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551364
proto_enc1_input+
proto_enc1_551305:	
proto_enc1_551307:	+
proto_enc1_551309:		
proto_enc1_551311:	+
proto_enc2_551314:	
proto_enc2_551316:+
proto_enc2_551318:
proto_enc2_551320:+
proto_enc3_551323:
proto_enc3_551325:+
proto_enc3_551327:
proto_enc3_551329:+
proto_dec3_551332:
proto_dec3_551334:+
proto_dec3_551336:
proto_dec3_551338:+
proto_dec2_551341:	
proto_dec2_551343:	+
proto_dec2_551345:		
proto_dec2_551347:	+
proto_dec1_551350:		
proto_dec1_551352:	+
proto_dec1_551354:		
proto_dec1_551356:	+
proto_dec1_551358:	
proto_dec1_551360:
identity¢"proto_dec1/StatefulPartitionedCall¢"proto_dec2/StatefulPartitionedCall¢"proto_dec3/StatefulPartitionedCall¢"proto_enc1/StatefulPartitionedCall¢"proto_enc2/StatefulPartitionedCall¢"proto_enc3/StatefulPartitionedCall¹
"proto_enc1/StatefulPartitionedCallStatefulPartitionedCallproto_enc1_inputproto_enc1_551305proto_enc1_551307proto_enc1_551309proto_enc1_551311*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549869Ô
"proto_enc2/StatefulPartitionedCallStatefulPartitionedCall+proto_enc1/StatefulPartitionedCall:output:0proto_enc2_551314proto_enc2_551316proto_enc2_551318proto_enc2_551320*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc2_layer_call_and_return_conditional_losses_550023Ô
"proto_enc3/StatefulPartitionedCallStatefulPartitionedCall+proto_enc2/StatefulPartitionedCall:output:0proto_enc3_551323proto_enc3_551325proto_enc3_551327proto_enc3_551329*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550177Ô
"proto_dec3/StatefulPartitionedCallStatefulPartitionedCall+proto_enc3/StatefulPartitionedCall:output:0proto_dec3_551332proto_dec3_551334proto_dec3_551336proto_dec3_551338*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550377Ô
"proto_dec2/StatefulPartitionedCallStatefulPartitionedCall+proto_dec3/StatefulPartitionedCall:output:0proto_dec2_551341proto_dec2_551343proto_dec2_551345proto_dec2_551347*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550577þ
"proto_dec1/StatefulPartitionedCallStatefulPartitionedCall+proto_dec2/StatefulPartitionedCall:output:0proto_dec1_551350proto_dec1_551352proto_dec1_551354proto_dec1_551356proto_dec1_551358proto_dec1_551360*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550816
IdentityIdentity+proto_dec1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¤
NoOpNoOp#^proto_dec1/StatefulPartitionedCall#^proto_dec2/StatefulPartitionedCall#^proto_dec3/StatefulPartitionedCall#^proto_enc1/StatefulPartitionedCall#^proto_enc2/StatefulPartitionedCall#^proto_enc3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿdd: : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"proto_dec1/StatefulPartitionedCall"proto_dec1/StatefulPartitionedCall2H
"proto_dec2/StatefulPartitionedCall"proto_dec2/StatefulPartitionedCall2H
"proto_dec3/StatefulPartitionedCall"proto_dec3/StatefulPartitionedCall2H
"proto_enc1/StatefulPartitionedCall"proto_enc1/StatefulPartitionedCall2H
"proto_enc2/StatefulPartitionedCall"proto_enc2/StatefulPartitionedCall2H
"proto_enc3/StatefulPartitionedCall"proto_enc3/StatefulPartitionedCall:a ]
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
*
_user_specified_nameproto_enc1_input
¤
ö%
__inference__traced_save_553116
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop8
4savev2_conv2d_transpose_4_kernel_read_readvariableop6
2savev2_conv2d_transpose_4_bias_read_readvariableop8
4savev2_conv2d_transpose_5_kernel_read_readvariableop6
2savev2_conv2d_transpose_5_bias_read_readvariableop8
4savev2_conv2d_transpose_2_kernel_read_readvariableop6
2savev2_conv2d_transpose_2_bias_read_readvariableop8
4savev2_conv2d_transpose_3_kernel_read_readvariableop6
2savev2_conv2d_transpose_3_bias_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop5
1savev2_adam_conv2d_1_kernel_m_read_readvariableop3
/savev2_adam_conv2d_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_3_kernel_m_read_readvariableop3
/savev2_adam_conv2d_3_bias_m_read_readvariableop5
1savev2_adam_conv2d_4_kernel_m_read_readvariableop3
/savev2_adam_conv2d_4_bias_m_read_readvariableop5
1savev2_adam_conv2d_5_kernel_m_read_readvariableop3
/savev2_adam_conv2d_5_bias_m_read_readvariableop5
1savev2_adam_conv2d_6_kernel_m_read_readvariableop3
/savev2_adam_conv2d_6_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_4_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_4_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_5_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_5_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_2_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_2_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_3_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_3_bias_m_read_readvariableop=
9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop;
7savev2_adam_conv2d_transpose_bias_m_read_readvariableop?
;savev2_adam_conv2d_transpose_1_kernel_m_read_readvariableop=
9savev2_adam_conv2d_transpose_1_bias_m_read_readvariableop5
1savev2_adam_conv2d_2_kernel_m_read_readvariableop3
/savev2_adam_conv2d_2_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop5
1savev2_adam_conv2d_1_kernel_v_read_readvariableop3
/savev2_adam_conv2d_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_3_kernel_v_read_readvariableop3
/savev2_adam_conv2d_3_bias_v_read_readvariableop5
1savev2_adam_conv2d_4_kernel_v_read_readvariableop3
/savev2_adam_conv2d_4_bias_v_read_readvariableop5
1savev2_adam_conv2d_5_kernel_v_read_readvariableop3
/savev2_adam_conv2d_5_bias_v_read_readvariableop5
1savev2_adam_conv2d_6_kernel_v_read_readvariableop3
/savev2_adam_conv2d_6_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_4_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_4_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_5_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_5_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_2_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_2_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_3_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_3_bias_v_read_readvariableop=
9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop;
7savev2_adam_conv2d_transpose_bias_v_read_readvariableop?
;savev2_adam_conv2d_transpose_1_kernel_v_read_readvariableop=
9savev2_adam_conv2d_transpose_1_bias_v_read_readvariableop5
1savev2_adam_conv2d_2_kernel_v_read_readvariableop3
/savev2_adam_conv2d_2_bias_v_read_readvariableop
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
: ×(
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*(
valueö'Bó'XB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/20/.ATTRIBUTES/VARIABLE_VALUEB'variables/21/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/19/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/20/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/21/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/22/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/23/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/24/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/25/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH 
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:X*
dtype0*Å
value»B¸XB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B µ$
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop4savev2_conv2d_transpose_4_kernel_read_readvariableop2savev2_conv2d_transpose_4_bias_read_readvariableop4savev2_conv2d_transpose_5_kernel_read_readvariableop2savev2_conv2d_transpose_5_bias_read_readvariableop4savev2_conv2d_transpose_2_kernel_read_readvariableop2savev2_conv2d_transpose_2_bias_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop1savev2_adam_conv2d_1_kernel_m_read_readvariableop/savev2_adam_conv2d_1_bias_m_read_readvariableop1savev2_adam_conv2d_3_kernel_m_read_readvariableop/savev2_adam_conv2d_3_bias_m_read_readvariableop1savev2_adam_conv2d_4_kernel_m_read_readvariableop/savev2_adam_conv2d_4_bias_m_read_readvariableop1savev2_adam_conv2d_5_kernel_m_read_readvariableop/savev2_adam_conv2d_5_bias_m_read_readvariableop1savev2_adam_conv2d_6_kernel_m_read_readvariableop/savev2_adam_conv2d_6_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_4_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_4_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_5_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_5_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_2_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_2_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_3_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_3_bias_m_read_readvariableop9savev2_adam_conv2d_transpose_kernel_m_read_readvariableop7savev2_adam_conv2d_transpose_bias_m_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_m_read_readvariableop9savev2_adam_conv2d_transpose_1_bias_m_read_readvariableop1savev2_adam_conv2d_2_kernel_m_read_readvariableop/savev2_adam_conv2d_2_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop1savev2_adam_conv2d_1_kernel_v_read_readvariableop/savev2_adam_conv2d_1_bias_v_read_readvariableop1savev2_adam_conv2d_3_kernel_v_read_readvariableop/savev2_adam_conv2d_3_bias_v_read_readvariableop1savev2_adam_conv2d_4_kernel_v_read_readvariableop/savev2_adam_conv2d_4_bias_v_read_readvariableop1savev2_adam_conv2d_5_kernel_v_read_readvariableop/savev2_adam_conv2d_5_bias_v_read_readvariableop1savev2_adam_conv2d_6_kernel_v_read_readvariableop/savev2_adam_conv2d_6_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_4_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_4_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_5_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_5_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_2_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_2_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_3_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_3_bias_v_read_readvariableop9savev2_adam_conv2d_transpose_kernel_v_read_readvariableop7savev2_adam_conv2d_transpose_bias_v_read_readvariableop;savev2_adam_conv2d_transpose_1_kernel_v_read_readvariableop9savev2_adam_conv2d_transpose_1_bias_v_read_readvariableop1savev2_adam_conv2d_2_kernel_v_read_readvariableop/savev2_adam_conv2d_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *f
dtypes\
Z2X	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
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

identity_1Identity_1:output:0*Ó
_input_shapesÁ
¾: :	:	:		:	:	::::::::::::	:	:		:	:		:	:		:	:	:: : : : : : : : : :	:	:		:	:	::::::::::::	:	:		:	:		:	:		:	:	::	:	:		:	:	::::::::::::	:	:		:	:		:	:		:	:	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:	: 

_output_shapes
:	:,(
&
_output_shapes
:		: 

_output_shapes
:	:,(
&
_output_shapes
:	: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:: 


_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:	: 

_output_shapes
:	:,(
&
_output_shapes
:		: 

_output_shapes
:	:,(
&
_output_shapes
:		: 

_output_shapes
:	:,(
&
_output_shapes
:		: 

_output_shapes
:	:,(
&
_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :,$(
&
_output_shapes
:	: %

_output_shapes
:	:,&(
&
_output_shapes
:		: '

_output_shapes
:	:,((
&
_output_shapes
:	: )

_output_shapes
::,*(
&
_output_shapes
:: +

_output_shapes
::,,(
&
_output_shapes
:: -

_output_shapes
::,.(
&
_output_shapes
:: /

_output_shapes
::,0(
&
_output_shapes
:: 1

_output_shapes
::,2(
&
_output_shapes
:: 3

_output_shapes
::,4(
&
_output_shapes
:	: 5

_output_shapes
:	:,6(
&
_output_shapes
:		: 7

_output_shapes
:	:,8(
&
_output_shapes
:		: 9

_output_shapes
:	:,:(
&
_output_shapes
:		: ;

_output_shapes
:	:,<(
&
_output_shapes
:	: =

_output_shapes
::,>(
&
_output_shapes
:	: ?

_output_shapes
:	:,@(
&
_output_shapes
:		: A

_output_shapes
:	:,B(
&
_output_shapes
:	: C

_output_shapes
::,D(
&
_output_shapes
:: E

_output_shapes
::,F(
&
_output_shapes
:: G

_output_shapes
::,H(
&
_output_shapes
:: I

_output_shapes
::,J(
&
_output_shapes
:: K

_output_shapes
::,L(
&
_output_shapes
:: M

_output_shapes
::,N(
&
_output_shapes
:	: O

_output_shapes
:	:,P(
&
_output_shapes
:		: Q

_output_shapes
:	:,R(
&
_output_shapes
:		: S

_output_shapes
:	:,T(
&
_output_shapes
:		: U

_output_shapes
:	:,V(
&
_output_shapes
:	: W

_output_shapes
::X

_output_shapes
: 
Ü)
È

K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551302
proto_enc1_input+
proto_enc1_551243:	
proto_enc1_551245:	+
proto_enc1_551247:		
proto_enc1_551249:	+
proto_enc2_551252:	
proto_enc2_551254:+
proto_enc2_551256:
proto_enc2_551258:+
proto_enc3_551261:
proto_enc3_551263:+
proto_enc3_551265:
proto_enc3_551267:+
proto_dec3_551270:
proto_dec3_551272:+
proto_dec3_551274:
proto_dec3_551276:+
proto_dec2_551279:	
proto_dec2_551281:	+
proto_dec2_551283:		
proto_dec2_551285:	+
proto_dec1_551288:		
proto_dec1_551290:	+
proto_dec1_551292:		
proto_dec1_551294:	+
proto_dec1_551296:	
proto_dec1_551298:
identity¢"proto_dec1/StatefulPartitionedCall¢"proto_dec2/StatefulPartitionedCall¢"proto_dec3/StatefulPartitionedCall¢"proto_enc1/StatefulPartitionedCall¢"proto_enc2/StatefulPartitionedCall¢"proto_enc3/StatefulPartitionedCall¹
"proto_enc1/StatefulPartitionedCallStatefulPartitionedCallproto_enc1_inputproto_enc1_551243proto_enc1_551245proto_enc1_551247proto_enc1_551249*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549809Ô
"proto_enc2/StatefulPartitionedCallStatefulPartitionedCall+proto_enc1/StatefulPartitionedCall:output:0proto_enc2_551252proto_enc2_551254proto_enc2_551256proto_enc2_551258*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc2_layer_call_and_return_conditional_losses_549963Ô
"proto_enc3/StatefulPartitionedCallStatefulPartitionedCall+proto_enc2/StatefulPartitionedCall:output:0proto_enc3_551261proto_enc3_551263proto_enc3_551265proto_enc3_551267*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550117Ô
"proto_dec3/StatefulPartitionedCallStatefulPartitionedCall+proto_enc3/StatefulPartitionedCall:output:0proto_dec3_551270proto_dec3_551272proto_dec3_551274proto_dec3_551276*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550337Ô
"proto_dec2/StatefulPartitionedCallStatefulPartitionedCall+proto_dec3/StatefulPartitionedCall:output:0proto_dec2_551279proto_dec2_551281proto_dec2_551283proto_dec2_551285*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550537þ
"proto_dec1/StatefulPartitionedCallStatefulPartitionedCall+proto_dec2/StatefulPartitionedCall:output:0proto_dec1_551288proto_dec1_551290proto_dec1_551292proto_dec1_551294proto_dec1_551296proto_dec1_551298*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550753
IdentityIdentity+proto_dec1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd¤
NoOpNoOp#^proto_dec1/StatefulPartitionedCall#^proto_dec2/StatefulPartitionedCall#^proto_dec3/StatefulPartitionedCall#^proto_enc1/StatefulPartitionedCall#^proto_enc2/StatefulPartitionedCall#^proto_enc3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿdd: : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"proto_dec1/StatefulPartitionedCall"proto_dec1/StatefulPartitionedCall2H
"proto_dec2/StatefulPartitionedCall"proto_dec2/StatefulPartitionedCall2H
"proto_dec3/StatefulPartitionedCall"proto_dec3/StatefulPartitionedCall2H
"proto_enc1/StatefulPartitionedCall"proto_enc1/StatefulPartitionedCall2H
"proto_enc2/StatefulPartitionedCall"proto_enc2/StatefulPartitionedCall2H
"proto_enc3/StatefulPartitionedCall"proto_enc3/StatefulPartitionedCall:a ]
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
*
_user_specified_nameproto_enc1_input

Ù
F__inference_proto_enc1_layer_call_and_return_conditional_losses_551949

inputs?
%conv2d_conv2d_readvariableop_resource:	4
&conv2d_biasadd_readvariableop_resource:	A
'conv2d_1_conv2d_readvariableop_resource:		6
(conv2d_1_biasadd_readvariableop_resource:	
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0§
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	d

conv2d/EluEluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0½
conv2d_1/Conv2DConv2Dconv2d/Elu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	h
conv2d_1/EluEluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	q
IdentityIdentityconv2d_1/Elu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	È
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿdd: : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs

¦
0__inference_proto_ae_123321_layer_call_fn_551007
proto_enc1_input!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:	
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:	

unknown_16:	$

unknown_17:		

unknown_18:	$

unknown_19:		

unknown_20:	$

unknown_21:		

unknown_22:	$

unknown_23:	

unknown_24:
identity¢StatefulPartitionedCall¾
StatefulPartitionedCallStatefulPartitionedCallproto_enc1_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_550952w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿdd: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:a ]
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
*
_user_specified_nameproto_enc1_input
ï

)__inference_conv2d_5_layer_call_fn_552524

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_550093w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Ò
ß
+__inference_proto_enc1_layer_call_fn_549820
input_1!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549809w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿdd: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
!
_user_specified_name	input_1
ë

'__inference_conv2d_layer_call_fn_552444

inputs!
unknown:	
	unknown_0:	
identity¢StatefulPartitionedCallä
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_549785w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿdd: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
þ

û
B__inference_conv2d_layer_call_and_return_conditional_losses_549785

inputs8
conv2d_readvariableop_resource:	-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿdd: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
¾
¤
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551715

inputsJ
0proto_enc1_conv2d_conv2d_readvariableop_resource:	?
1proto_enc1_conv2d_biasadd_readvariableop_resource:	L
2proto_enc1_conv2d_1_conv2d_readvariableop_resource:		A
3proto_enc1_conv2d_1_biasadd_readvariableop_resource:	L
2proto_enc2_conv2d_3_conv2d_readvariableop_resource:	A
3proto_enc2_conv2d_3_biasadd_readvariableop_resource:L
2proto_enc2_conv2d_4_conv2d_readvariableop_resource:A
3proto_enc2_conv2d_4_biasadd_readvariableop_resource:L
2proto_enc3_conv2d_5_conv2d_readvariableop_resource:A
3proto_enc3_conv2d_5_biasadd_readvariableop_resource:L
2proto_enc3_conv2d_6_conv2d_readvariableop_resource:A
3proto_enc3_conv2d_6_biasadd_readvariableop_resource:`
Fproto_dec3_conv2d_transpose_4_conv2d_transpose_readvariableop_resource:K
=proto_dec3_conv2d_transpose_4_biasadd_readvariableop_resource:`
Fproto_dec3_conv2d_transpose_5_conv2d_transpose_readvariableop_resource:K
=proto_dec3_conv2d_transpose_5_biasadd_readvariableop_resource:`
Fproto_dec2_conv2d_transpose_2_conv2d_transpose_readvariableop_resource:	K
=proto_dec2_conv2d_transpose_2_biasadd_readvariableop_resource:	`
Fproto_dec2_conv2d_transpose_3_conv2d_transpose_readvariableop_resource:		K
=proto_dec2_conv2d_transpose_3_biasadd_readvariableop_resource:	^
Dproto_dec1_conv2d_transpose_conv2d_transpose_readvariableop_resource:		I
;proto_dec1_conv2d_transpose_biasadd_readvariableop_resource:	`
Fproto_dec1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource:		K
=proto_dec1_conv2d_transpose_1_biasadd_readvariableop_resource:	L
2proto_dec1_conv2d_2_conv2d_readvariableop_resource:	A
3proto_dec1_conv2d_2_biasadd_readvariableop_resource:
identity¢*proto_dec1/conv2d_2/BiasAdd/ReadVariableOp¢)proto_dec1/conv2d_2/Conv2D/ReadVariableOp¢2proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOp¢;proto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOp¢4proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOp¢=proto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp¢4proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOp¢=proto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp¢4proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOp¢=proto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp¢4proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOp¢=proto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOp¢4proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOp¢=proto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOp¢(proto_enc1/conv2d/BiasAdd/ReadVariableOp¢'proto_enc1/conv2d/Conv2D/ReadVariableOp¢*proto_enc1/conv2d_1/BiasAdd/ReadVariableOp¢)proto_enc1/conv2d_1/Conv2D/ReadVariableOp¢*proto_enc2/conv2d_3/BiasAdd/ReadVariableOp¢)proto_enc2/conv2d_3/Conv2D/ReadVariableOp¢*proto_enc2/conv2d_4/BiasAdd/ReadVariableOp¢)proto_enc2/conv2d_4/Conv2D/ReadVariableOp¢*proto_enc3/conv2d_5/BiasAdd/ReadVariableOp¢)proto_enc3/conv2d_5/Conv2D/ReadVariableOp¢*proto_enc3/conv2d_6/BiasAdd/ReadVariableOp¢)proto_enc3/conv2d_6/Conv2D/ReadVariableOp 
'proto_enc1/conv2d/Conv2D/ReadVariableOpReadVariableOp0proto_enc1_conv2d_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0½
proto_enc1/conv2d/Conv2DConv2Dinputs/proto_enc1/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides

(proto_enc1/conv2d/BiasAdd/ReadVariableOpReadVariableOp1proto_enc1_conv2d_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0³
proto_enc1/conv2d/BiasAddBiasAdd!proto_enc1/conv2d/Conv2D:output:00proto_enc1/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	z
proto_enc1/conv2d/EluElu"proto_enc1/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	¤
)proto_enc1/conv2d_1/Conv2D/ReadVariableOpReadVariableOp2proto_enc1_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0Þ
proto_enc1/conv2d_1/Conv2DConv2D#proto_enc1/conv2d/Elu:activations:01proto_enc1/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides

*proto_enc1/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp3proto_enc1_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0¹
proto_enc1/conv2d_1/BiasAddBiasAdd#proto_enc1/conv2d_1/Conv2D:output:02proto_enc1/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	~
proto_enc1/conv2d_1/EluElu$proto_enc1/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	¤
)proto_enc2/conv2d_3/Conv2D/ReadVariableOpReadVariableOp2proto_enc2_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0à
proto_enc2/conv2d_3/Conv2DConv2D%proto_enc1/conv2d_1/Elu:activations:01proto_enc2/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

*proto_enc2/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp3proto_enc2_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¹
proto_enc2/conv2d_3/BiasAddBiasAdd#proto_enc2/conv2d_3/Conv2D:output:02proto_enc2/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22~
proto_enc2/conv2d_3/EluElu$proto_enc2/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
)proto_enc2/conv2d_4/Conv2D/ReadVariableOpReadVariableOp2proto_enc2_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0à
proto_enc2/conv2d_4/Conv2DConv2D%proto_enc2/conv2d_3/Elu:activations:01proto_enc2/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

*proto_enc2/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp3proto_enc2_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¹
proto_enc2/conv2d_4/BiasAddBiasAdd#proto_enc2/conv2d_4/Conv2D:output:02proto_enc2/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22~
proto_enc2/conv2d_4/EluElu$proto_enc2/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
)proto_enc3/conv2d_5/Conv2D/ReadVariableOpReadVariableOp2proto_enc3_conv2d_5_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0à
proto_enc3/conv2d_5/Conv2DConv2D%proto_enc2/conv2d_4/Elu:activations:01proto_enc3/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

*proto_enc3/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp3proto_enc3_conv2d_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¹
proto_enc3/conv2d_5/BiasAddBiasAdd#proto_enc3/conv2d_5/Conv2D:output:02proto_enc3/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22~
proto_enc3/conv2d_5/EluElu$proto_enc3/conv2d_5/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22¤
)proto_enc3/conv2d_6/Conv2D/ReadVariableOpReadVariableOp2proto_enc3_conv2d_6_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype0à
proto_enc3/conv2d_6/Conv2DConv2D%proto_enc3/conv2d_5/Elu:activations:01proto_enc3/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides

*proto_enc3/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp3proto_enc3_conv2d_6_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¹
proto_enc3/conv2d_6/BiasAddBiasAdd#proto_enc3/conv2d_6/Conv2D:output:02proto_enc3/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22~
proto_enc3/conv2d_6/EluElu$proto_enc3/conv2d_6/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22x
#proto_dec3/conv2d_transpose_4/ShapeShape%proto_enc3/conv2d_6/Elu:activations:0*
T0*
_output_shapes
:{
1proto_dec3/conv2d_transpose_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3proto_dec3/conv2d_transpose_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3proto_dec3/conv2d_transpose_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+proto_dec3/conv2d_transpose_4/strided_sliceStridedSlice,proto_dec3/conv2d_transpose_4/Shape:output:0:proto_dec3/conv2d_transpose_4/strided_slice/stack:output:0<proto_dec3/conv2d_transpose_4/strided_slice/stack_1:output:0<proto_dec3/conv2d_transpose_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%proto_dec3/conv2d_transpose_4/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2g
%proto_dec3/conv2d_transpose_4/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2g
%proto_dec3/conv2d_transpose_4/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
#proto_dec3/conv2d_transpose_4/stackPack4proto_dec3/conv2d_transpose_4/strided_slice:output:0.proto_dec3/conv2d_transpose_4/stack/1:output:0.proto_dec3/conv2d_transpose_4/stack/2:output:0.proto_dec3/conv2d_transpose_4/stack/3:output:0*
N*
T0*
_output_shapes
:}
3proto_dec3/conv2d_transpose_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5proto_dec3/conv2d_transpose_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5proto_dec3/conv2d_transpose_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
-proto_dec3/conv2d_transpose_4/strided_slice_1StridedSlice,proto_dec3/conv2d_transpose_4/stack:output:0<proto_dec3/conv2d_transpose_4/strided_slice_1/stack:output:0>proto_dec3/conv2d_transpose_4/strided_slice_1/stack_1:output:0>proto_dec3/conv2d_transpose_4/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÌ
=proto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOpReadVariableOpFproto_dec3_conv2d_transpose_4_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0Ã
.proto_dec3/conv2d_transpose_4/conv2d_transposeConv2DBackpropInput,proto_dec3/conv2d_transpose_4/stack:output:0Eproto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOp:value:0%proto_enc3/conv2d_6/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
®
4proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOpReadVariableOp=proto_dec3_conv2d_transpose_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0á
%proto_dec3/conv2d_transpose_4/BiasAddBiasAdd7proto_dec3/conv2d_transpose_4/conv2d_transpose:output:0<proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!proto_dec3/conv2d_transpose_4/EluElu.proto_dec3/conv2d_transpose_4/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
#proto_dec3/conv2d_transpose_5/ShapeShape/proto_dec3/conv2d_transpose_4/Elu:activations:0*
T0*
_output_shapes
:{
1proto_dec3/conv2d_transpose_5/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3proto_dec3/conv2d_transpose_5/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3proto_dec3/conv2d_transpose_5/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+proto_dec3/conv2d_transpose_5/strided_sliceStridedSlice,proto_dec3/conv2d_transpose_5/Shape:output:0:proto_dec3/conv2d_transpose_5/strided_slice/stack:output:0<proto_dec3/conv2d_transpose_5/strided_slice/stack_1:output:0<proto_dec3/conv2d_transpose_5/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%proto_dec3/conv2d_transpose_5/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2g
%proto_dec3/conv2d_transpose_5/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2g
%proto_dec3/conv2d_transpose_5/stack/3Const*
_output_shapes
: *
dtype0*
value	B :
#proto_dec3/conv2d_transpose_5/stackPack4proto_dec3/conv2d_transpose_5/strided_slice:output:0.proto_dec3/conv2d_transpose_5/stack/1:output:0.proto_dec3/conv2d_transpose_5/stack/2:output:0.proto_dec3/conv2d_transpose_5/stack/3:output:0*
N*
T0*
_output_shapes
:}
3proto_dec3/conv2d_transpose_5/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5proto_dec3/conv2d_transpose_5/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5proto_dec3/conv2d_transpose_5/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
-proto_dec3/conv2d_transpose_5/strided_slice_1StridedSlice,proto_dec3/conv2d_transpose_5/stack:output:0<proto_dec3/conv2d_transpose_5/strided_slice_1/stack:output:0>proto_dec3/conv2d_transpose_5/strided_slice_1/stack_1:output:0>proto_dec3/conv2d_transpose_5/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÌ
=proto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOpReadVariableOpFproto_dec3_conv2d_transpose_5_conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0Í
.proto_dec3/conv2d_transpose_5/conv2d_transposeConv2DBackpropInput,proto_dec3/conv2d_transpose_5/stack:output:0Eproto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOp:value:0/proto_dec3/conv2d_transpose_4/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
®
4proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOpReadVariableOp=proto_dec3_conv2d_transpose_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0á
%proto_dec3/conv2d_transpose_5/BiasAddBiasAdd7proto_dec3/conv2d_transpose_5/conv2d_transpose:output:0<proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!proto_dec3/conv2d_transpose_5/EluElu.proto_dec3/conv2d_transpose_5/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
#proto_dec2/conv2d_transpose_2/ShapeShape/proto_dec3/conv2d_transpose_5/Elu:activations:0*
T0*
_output_shapes
:{
1proto_dec2/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3proto_dec2/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3proto_dec2/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+proto_dec2/conv2d_transpose_2/strided_sliceStridedSlice,proto_dec2/conv2d_transpose_2/Shape:output:0:proto_dec2/conv2d_transpose_2/strided_slice/stack:output:0<proto_dec2/conv2d_transpose_2/strided_slice/stack_1:output:0<proto_dec2/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%proto_dec2/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2g
%proto_dec2/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2g
%proto_dec2/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	
#proto_dec2/conv2d_transpose_2/stackPack4proto_dec2/conv2d_transpose_2/strided_slice:output:0.proto_dec2/conv2d_transpose_2/stack/1:output:0.proto_dec2/conv2d_transpose_2/stack/2:output:0.proto_dec2/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:}
3proto_dec2/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5proto_dec2/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5proto_dec2/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
-proto_dec2/conv2d_transpose_2/strided_slice_1StridedSlice,proto_dec2/conv2d_transpose_2/stack:output:0<proto_dec2/conv2d_transpose_2/strided_slice_1/stack:output:0>proto_dec2/conv2d_transpose_2/strided_slice_1/stack_1:output:0>proto_dec2/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÌ
=proto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpFproto_dec2_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:	*
dtype0Í
.proto_dec2/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput,proto_dec2/conv2d_transpose_2/stack:output:0Eproto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0/proto_dec3/conv2d_transpose_5/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides
®
4proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp=proto_dec2_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0á
%proto_dec2/conv2d_transpose_2/BiasAddBiasAdd7proto_dec2/conv2d_transpose_2/conv2d_transpose:output:0<proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
!proto_dec2/conv2d_transpose_2/EluElu.proto_dec2/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
#proto_dec2/conv2d_transpose_3/ShapeShape/proto_dec2/conv2d_transpose_2/Elu:activations:0*
T0*
_output_shapes
:{
1proto_dec2/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3proto_dec2/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3proto_dec2/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+proto_dec2/conv2d_transpose_3/strided_sliceStridedSlice,proto_dec2/conv2d_transpose_3/Shape:output:0:proto_dec2/conv2d_transpose_3/strided_slice/stack:output:0<proto_dec2/conv2d_transpose_3/strided_slice/stack_1:output:0<proto_dec2/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%proto_dec2/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2g
%proto_dec2/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2g
%proto_dec2/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	
#proto_dec2/conv2d_transpose_3/stackPack4proto_dec2/conv2d_transpose_3/strided_slice:output:0.proto_dec2/conv2d_transpose_3/stack/1:output:0.proto_dec2/conv2d_transpose_3/stack/2:output:0.proto_dec2/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:}
3proto_dec2/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5proto_dec2/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5proto_dec2/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
-proto_dec2/conv2d_transpose_3/strided_slice_1StridedSlice,proto_dec2/conv2d_transpose_3/stack:output:0<proto_dec2/conv2d_transpose_3/strided_slice_1/stack:output:0>proto_dec2/conv2d_transpose_3/strided_slice_1/stack_1:output:0>proto_dec2/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÌ
=proto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpFproto_dec2_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0Í
.proto_dec2/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput,proto_dec2/conv2d_transpose_3/stack:output:0Eproto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0/proto_dec2/conv2d_transpose_2/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides
®
4proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp=proto_dec2_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0á
%proto_dec2/conv2d_transpose_3/BiasAddBiasAdd7proto_dec2/conv2d_transpose_3/conv2d_transpose:output:0<proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
!proto_dec2/conv2d_transpose_3/EluElu.proto_dec2/conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
!proto_dec1/conv2d_transpose/ShapeShape/proto_dec2/conv2d_transpose_3/Elu:activations:0*
T0*
_output_shapes
:y
/proto_dec1/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: {
1proto_dec1/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:{
1proto_dec1/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ý
)proto_dec1/conv2d_transpose/strided_sliceStridedSlice*proto_dec1/conv2d_transpose/Shape:output:08proto_dec1/conv2d_transpose/strided_slice/stack:output:0:proto_dec1/conv2d_transpose/strided_slice/stack_1:output:0:proto_dec1/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maske
#proto_dec1/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2e
#proto_dec1/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2e
#proto_dec1/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	
!proto_dec1/conv2d_transpose/stackPack2proto_dec1/conv2d_transpose/strided_slice:output:0,proto_dec1/conv2d_transpose/stack/1:output:0,proto_dec1/conv2d_transpose/stack/2:output:0,proto_dec1/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:{
1proto_dec1/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3proto_dec1/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3proto_dec1/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:å
+proto_dec1/conv2d_transpose/strided_slice_1StridedSlice*proto_dec1/conv2d_transpose/stack:output:0:proto_dec1/conv2d_transpose/strided_slice_1/stack:output:0<proto_dec1/conv2d_transpose/strided_slice_1/stack_1:output:0<proto_dec1/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÈ
;proto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpDproto_dec1_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0Ç
,proto_dec1/conv2d_transpose/conv2d_transposeConv2DBackpropInput*proto_dec1/conv2d_transpose/stack:output:0Cproto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0/proto_dec2/conv2d_transpose_3/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides
ª
2proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp;proto_dec1_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0Û
#proto_dec1/conv2d_transpose/BiasAddBiasAdd5proto_dec1/conv2d_transpose/conv2d_transpose:output:0:proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
proto_dec1/conv2d_transpose/EluElu,proto_dec1/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
#proto_dec1/conv2d_transpose_1/ShapeShape-proto_dec1/conv2d_transpose/Elu:activations:0*
T0*
_output_shapes
:{
1proto_dec1/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: }
3proto_dec1/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:}
3proto_dec1/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ç
+proto_dec1/conv2d_transpose_1/strided_sliceStridedSlice,proto_dec1/conv2d_transpose_1/Shape:output:0:proto_dec1/conv2d_transpose_1/strided_slice/stack:output:0<proto_dec1/conv2d_transpose_1/strided_slice/stack_1:output:0<proto_dec1/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskg
%proto_dec1/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :dg
%proto_dec1/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :dg
%proto_dec1/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	
#proto_dec1/conv2d_transpose_1/stackPack4proto_dec1/conv2d_transpose_1/strided_slice:output:0.proto_dec1/conv2d_transpose_1/stack/1:output:0.proto_dec1/conv2d_transpose_1/stack/2:output:0.proto_dec1/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:}
3proto_dec1/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 
5proto_dec1/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:
5proto_dec1/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:ï
-proto_dec1/conv2d_transpose_1/strided_slice_1StridedSlice,proto_dec1/conv2d_transpose_1/stack:output:0<proto_dec1/conv2d_transpose_1/strided_slice_1/stack:output:0>proto_dec1/conv2d_transpose_1/strided_slice_1/stack_1:output:0>proto_dec1/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskÌ
=proto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpFproto_dec1_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0Ë
.proto_dec1/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput,proto_dec1/conv2d_transpose_1/stack:output:0Eproto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0-proto_dec1/conv2d_transpose/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	*
paddingSAME*
strides
®
4proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp=proto_dec1_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0á
%proto_dec1/conv2d_transpose_1/BiasAddBiasAdd7proto_dec1/conv2d_transpose_1/conv2d_transpose:output:0<proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	
!proto_dec1/conv2d_transpose_1/EluElu.proto_dec1/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	¤
)proto_dec1/conv2d_2/Conv2D/ReadVariableOpReadVariableOp2proto_dec1_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0ë
proto_dec1/conv2d_2/Conv2DConv2D/proto_dec1/conv2d_transpose_1/Elu:activations:01proto_dec1/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingVALID*
strides

*proto_dec1/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp3proto_dec1_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¹
proto_dec1/conv2d_2/BiasAddBiasAdd#proto_dec1/conv2d_2/Conv2D:output:02proto_dec1/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd{
IdentityIdentity$proto_dec1/conv2d_2/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd÷

NoOpNoOp+^proto_dec1/conv2d_2/BiasAdd/ReadVariableOp*^proto_dec1/conv2d_2/Conv2D/ReadVariableOp3^proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOp<^proto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOp5^proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOp>^proto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp5^proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOp>^proto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp5^proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOp>^proto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp5^proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOp>^proto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOp5^proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOp>^proto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOp)^proto_enc1/conv2d/BiasAdd/ReadVariableOp(^proto_enc1/conv2d/Conv2D/ReadVariableOp+^proto_enc1/conv2d_1/BiasAdd/ReadVariableOp*^proto_enc1/conv2d_1/Conv2D/ReadVariableOp+^proto_enc2/conv2d_3/BiasAdd/ReadVariableOp*^proto_enc2/conv2d_3/Conv2D/ReadVariableOp+^proto_enc2/conv2d_4/BiasAdd/ReadVariableOp*^proto_enc2/conv2d_4/Conv2D/ReadVariableOp+^proto_enc3/conv2d_5/BiasAdd/ReadVariableOp*^proto_enc3/conv2d_5/Conv2D/ReadVariableOp+^proto_enc3/conv2d_6/BiasAdd/ReadVariableOp*^proto_enc3/conv2d_6/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿdd: : : : : : : : : : : : : : : : : : : : : : : : : : 2X
*proto_dec1/conv2d_2/BiasAdd/ReadVariableOp*proto_dec1/conv2d_2/BiasAdd/ReadVariableOp2V
)proto_dec1/conv2d_2/Conv2D/ReadVariableOp)proto_dec1/conv2d_2/Conv2D/ReadVariableOp2h
2proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOp2proto_dec1/conv2d_transpose/BiasAdd/ReadVariableOp2z
;proto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOp;proto_dec1/conv2d_transpose/conv2d_transpose/ReadVariableOp2l
4proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOp4proto_dec1/conv2d_transpose_1/BiasAdd/ReadVariableOp2~
=proto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp=proto_dec1/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2l
4proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOp4proto_dec2/conv2d_transpose_2/BiasAdd/ReadVariableOp2~
=proto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp=proto_dec2/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2l
4proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOp4proto_dec2/conv2d_transpose_3/BiasAdd/ReadVariableOp2~
=proto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp=proto_dec2/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2l
4proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOp4proto_dec3/conv2d_transpose_4/BiasAdd/ReadVariableOp2~
=proto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOp=proto_dec3/conv2d_transpose_4/conv2d_transpose/ReadVariableOp2l
4proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOp4proto_dec3/conv2d_transpose_5/BiasAdd/ReadVariableOp2~
=proto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOp=proto_dec3/conv2d_transpose_5/conv2d_transpose/ReadVariableOp2T
(proto_enc1/conv2d/BiasAdd/ReadVariableOp(proto_enc1/conv2d/BiasAdd/ReadVariableOp2R
'proto_enc1/conv2d/Conv2D/ReadVariableOp'proto_enc1/conv2d/Conv2D/ReadVariableOp2X
*proto_enc1/conv2d_1/BiasAdd/ReadVariableOp*proto_enc1/conv2d_1/BiasAdd/ReadVariableOp2V
)proto_enc1/conv2d_1/Conv2D/ReadVariableOp)proto_enc1/conv2d_1/Conv2D/ReadVariableOp2X
*proto_enc2/conv2d_3/BiasAdd/ReadVariableOp*proto_enc2/conv2d_3/BiasAdd/ReadVariableOp2V
)proto_enc2/conv2d_3/Conv2D/ReadVariableOp)proto_enc2/conv2d_3/Conv2D/ReadVariableOp2X
*proto_enc2/conv2d_4/BiasAdd/ReadVariableOp*proto_enc2/conv2d_4/BiasAdd/ReadVariableOp2V
)proto_enc2/conv2d_4/Conv2D/ReadVariableOp)proto_enc2/conv2d_4/Conv2D/ReadVariableOp2X
*proto_enc3/conv2d_5/BiasAdd/ReadVariableOp*proto_enc3/conv2d_5/BiasAdd/ReadVariableOp2V
)proto_enc3/conv2d_5/Conv2D/ReadVariableOp)proto_enc3/conv2d_5/Conv2D/ReadVariableOp2X
*proto_enc3/conv2d_6/BiasAdd/ReadVariableOp*proto_enc3/conv2d_6/BiasAdd/ReadVariableOp2V
)proto_enc3/conv2d_6/Conv2D/ReadVariableOp)proto_enc3/conv2d_6/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
Ì
¨
3__inference_conv2d_transpose_1_layer_call_fn_552779

inputs!
unknown:		
	unknown_0:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_550712
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
¿!

L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_552770

inputsB
(conv2d_transpose_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :	y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
Ì
¨
3__inference_conv2d_transpose_4_layer_call_fn_552564

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_550267
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¼
ü
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550629
input_43
conv2d_transpose_2_550618:	'
conv2d_transpose_2_550620:	3
conv2d_transpose_3_550623:		'
conv2d_transpose_3_550625:	
identity¢*conv2d_transpose_2/StatefulPartitionedCall¢*conv2d_transpose_3/StatefulPartitionedCall¦
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCallinput_4conv2d_transpose_2_550618conv2d_transpose_2_550620*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_550467Ò
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0conv2d_transpose_3_550623conv2d_transpose_3_550625*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_550512
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	 
NoOpNoOp+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input_4
®9
Ù
F__inference_proto_dec2_layer_call_and_return_conditional_losses_552301

inputsU
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource:	@
2conv2d_transpose_2_biasadd_readvariableop_resource:	U
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource:		@
2conv2d_transpose_3_biasadd_readvariableop_resource:	
identity¢)conv2d_transpose_2/BiasAdd/ReadVariableOp¢2conv2d_transpose_2/conv2d_transpose/ReadVariableOp¢)conv2d_transpose_3/BiasAdd/ReadVariableOp¢2conv2d_transpose_3/conv2d_transpose/ReadVariableOpN
conv2d_transpose_2/ShapeShapeinputs*
T0*
_output_shapes
:p
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2\
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2\
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	è
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¶
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
:	*
dtype0
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides

)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0À
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	|
conv2d_transpose_2/EluElu#conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	l
conv2d_transpose_3/ShapeShape$conv2d_transpose_2/Elu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2\
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2\
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	è
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¶
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0¡
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0$conv2d_transpose_2/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides

)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0À
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	|
conv2d_transpose_3/EluElu#conv2d_transpose_3/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	{
IdentityIdentity$conv2d_transpose_3/Elu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
NoOpNoOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Ç
¹
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549809

inputs'
conv2d_549786:	
conv2d_549788:	)
conv2d_1_549803:		
conv2d_1_549805:	
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCallõ
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_549786conv2d_549788*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_549785
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_549803conv2d_1_549805*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_549802
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿdd: : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
Ì@

F__inference_proto_dec1_layer_call_and_return_conditional_losses_552385

inputsS
9conv2d_transpose_conv2d_transpose_readvariableop_resource:		>
0conv2d_transpose_biasadd_readvariableop_resource:	U
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource:		@
2conv2d_transpose_1_biasadd_readvariableop_resource:	A
'conv2d_2_conv2d_readvariableop_resource:	6
(conv2d_2_biasadd_readvariableop_resource:
identity¢conv2d_2/BiasAdd/ReadVariableOp¢conv2d_2/Conv2D/ReadVariableOp¢'conv2d_transpose/BiasAdd/ReadVariableOp¢0conv2d_transpose/conv2d_transpose/ReadVariableOp¢)conv2d_transpose_1/BiasAdd/ReadVariableOp¢2conv2d_transpose_1/conv2d_transpose/ReadVariableOpL
conv2d_transpose/ShapeShapeinputs*
T0*
_output_shapes
:n
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: p
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:p
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¦
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskZ
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2Z
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2Z
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	Þ
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:p
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:®
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask²
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0ý
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0inputs*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides

'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0º
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	x
conv2d_transpose/EluElu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	j
conv2d_transpose_1/ShapeShape"conv2d_transpose/Elu:activations:0*
T0*
_output_shapes
:p
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: r
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:r
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:°
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask\
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :d\
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :d\
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B :	è
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:r
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: t
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:t
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:¸
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask¶
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0"conv2d_transpose/Elu:activations:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	*
paddingSAME*
strides

)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0À
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	|
conv2d_transpose_1/EluElu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd	
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0Ê
conv2d_2/Conv2DConv2D$conv2d_transpose_1/Elu:activations:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*
paddingVALID*
strides

conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿddp
IdentityIdentityconv2d_2/BiasAdd:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿddÇ
NoOpNoOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ22	: : : : : : 2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs
Ò
ß
+__inference_proto_enc2_layer_call_fn_550047
input_3!
unknown:	
	unknown_0:#
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_3unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc2_layer_call_and_return_conditional_losses_550023w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
!
_user_specified_name	input_3
Ï
Þ
+__inference_proto_enc2_layer_call_fn_551975

inputs!
unknown:	
	unknown_0:#
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc2_layer_call_and_return_conditional_losses_550023w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22	: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs
Ê
º
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549921
input_1'
conv2d_549910:	
conv2d_549912:	)
conv2d_1_549915:		
conv2d_1_549917:	
identity¢conv2d/StatefulPartitionedCall¢ conv2d_1/StatefulPartitionedCallö
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_549910conv2d_549912*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *K
fFRD
B__inference_conv2d_layer_call_and_return_conditional_losses_549785
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0conv2d_1_549915conv2d_1_549917*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_1_layer_call_and_return_conditional_losses_549802
IdentityIdentity)conv2d_1/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
NoOpNoOp^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿdd: : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
!
_user_specified_name	input_1
ë

0__inference_proto_ae_123321_layer_call_fn_551486

inputs!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:	
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:
	unknown_8:#
	unknown_9:

unknown_10:$

unknown_11:

unknown_12:$

unknown_13:

unknown_14:$

unknown_15:	

unknown_16:	$

unknown_17:		

unknown_18:	$

unknown_19:		

unknown_20:	$

unknown_21:		

unknown_22:	$

unknown_23:	

unknown_24:
identity¢StatefulPartitionedCall´
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*<
_read_only_resource_inputs
	
*2
config_proto" 

CPU

GPU2 *0J 8 *T
fORM
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_550952w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿdd: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs

ý
D__inference_conv2d_4_layer_call_and_return_conditional_losses_549956

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Á!

N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_552641

inputsB
(conv2d_transpose_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿh
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿz
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ý
D__inference_conv2d_6_layer_call_and_return_conditional_losses_550110

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22V
EluEluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22h
IdentityIdentityElu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Ò
ß
+__inference_proto_enc1_layer_call_fn_549893
input_1!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549869w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿdd: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
!
_user_specified_name	input_1
Ï
Þ
+__inference_proto_enc1_layer_call_fn_551900

inputs!
unknown:	
	unknown_0:	#
	unknown_1:		
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549809w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿdd: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
¿!

L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_550667

inputsB
(conv2d_transpose_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :	y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs

Ù
F__inference_proto_enc1_layer_call_and_return_conditional_losses_551931

inputs?
%conv2d_conv2d_readvariableop_resource:	4
&conv2d_biasadd_readvariableop_resource:	A
'conv2d_1_conv2d_readvariableop_resource:		6
(conv2d_1_biasadd_readvariableop_resource:	
identity¢conv2d/BiasAdd/ReadVariableOp¢conv2d/Conv2D/ReadVariableOp¢conv2d_1/BiasAdd/ReadVariableOp¢conv2d_1/Conv2D/ReadVariableOp
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
:	*
dtype0§
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides

conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	d

conv2d/EluEluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:		*
dtype0½
conv2d_1/Conv2DConv2Dconv2d/Elu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	*
paddingSAME*
strides

conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	h
conv2d_1/EluEluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	q
IdentityIdentityconv2d_1/Elu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	È
NoOpNoOp^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿdd: : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd
 
_user_specified_nameinputs
¼
ü
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550415
input_63
conv2d_transpose_4_550404:'
conv2d_transpose_4_550406:3
conv2d_transpose_5_550409:'
conv2d_transpose_5_550411:
identity¢*conv2d_transpose_4/StatefulPartitionedCall¢*conv2d_transpose_5/StatefulPartitionedCall¦
*conv2d_transpose_4/StatefulPartitionedCallStatefulPartitionedCallinput_6conv2d_transpose_4_550404conv2d_transpose_4_550406*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_550267Ò
*conv2d_transpose_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_4/StatefulPartitionedCall:output:0conv2d_transpose_5_550409conv2d_transpose_5_550411*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *W
fRRP
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_550312
IdentityIdentity3conv2d_transpose_5/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22 
NoOpNoOp+^conv2d_transpose_4/StatefulPartitionedCall+^conv2d_transpose_5/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2X
*conv2d_transpose_4/StatefulPartitionedCall*conv2d_transpose_4/StatefulPartitionedCall2X
*conv2d_transpose_5/StatefulPartitionedCall*conv2d_transpose_5/StatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input_6
Á!

N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_552727

inputsB
(conv2d_transpose_readvariableop_resource:		-
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢conv2d_transpose/ReadVariableOp;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ñ
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskG
mul/yConst*
_output_shapes
: *
dtype0*
value	B :U
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: I
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :Y
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: I
stack/3Const*
_output_shapes
: *
dtype0*
value	B :	y
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:_
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: a
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:a
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:Ù
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:		*
dtype0Ü
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	*
paddingSAME*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:	*
dtype0
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	h
EluEluBiasAdd:output:0*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	z
IdentityIdentityElu:activations:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
NoOpNoOp^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
 
_user_specified_nameinputs
Ò
ß
+__inference_proto_enc3_layer_call_fn_550201
input_5!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_5unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550177w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input_5
«	

+__inference_proto_dec1_layer_call_fn_552335

inputs!
unknown:		
	unknown_0:	#
	unknown_1:		
	unknown_2:	#
	unknown_3:	
	unknown_4:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550816w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿdd`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':ÿÿÿÿÿÿÿÿÿ22	: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22	
 
_user_specified_nameinputs
ï

)__inference_conv2d_6_layer_call_fn_552544

inputs!
unknown:
	unknown_0:
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_550110w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ22: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Ý
¿
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550177

inputs)
conv2d_5_550166:
conv2d_5_550168:)
conv2d_6_550171:
conv2d_6_550173:
identity¢ conv2d_5/StatefulPartitionedCall¢ conv2d_6/StatefulPartitionedCallý
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_5_550166conv2d_5_550168*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_5_layer_call_and_return_conditional_losses_550093 
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0conv2d_6_550171conv2d_6_550173*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *M
fHRF
D__inference_conv2d_6_layer_call_and_return_conditional_losses_550110
IdentityIdentity)conv2d_6/StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
NoOpNoOp!^conv2d_5/StatefulPartitionedCall!^conv2d_6/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
 
_user_specified_nameinputs
Ò
ß
+__inference_proto_dec3_layer_call_fn_550348
input_6!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_6unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22*&
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2 *0J 8 *O
fJRH
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550337w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:ÿÿÿÿÿÿÿÿÿ22: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
!
_user_specified_name	input_6"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ï
serving_default»
U
proto_enc1_inputA
"serving_default_proto_enc1_input:0ÿÿÿÿÿÿÿÿÿddF

proto_dec18
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿddtensorflow/serving/predict:ç
Ð
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	variables
trainable_variables
	regularization_losses

	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
§
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
#_self_saveable_object_factories"
_tf_keras_network
§
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
trainable_variables
regularization_losses
 	keras_api
!__call__
*"&call_and_return_all_conditional_losses
##_self_saveable_object_factories"
_tf_keras_network
§
$layer-0
%layer_with_weights-0
%layer-1
&layer_with_weights-1
&layer-2
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses
#-_self_saveable_object_factories"
_tf_keras_network
§
.layer-0
/layer_with_weights-0
/layer-1
0layer_with_weights-1
0layer-2
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses
#7_self_saveable_object_factories"
_tf_keras_network
§
8layer-0
9layer_with_weights-0
9layer-1
:layer_with_weights-1
:layer-2
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses
#A_self_saveable_object_factories"
_tf_keras_network
Î
Blayer-0
Clayer_with_weights-0
Clayer-1
Dlayer_with_weights-1
Dlayer-2
Elayer_with_weights-2
Elayer-3
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
J__call__
*K&call_and_return_all_conditional_losses
#L_self_saveable_object_factories"
_tf_keras_network
æ
M0
N1
O2
P3
Q4
R5
S6
T7
U8
V9
W10
X11
Y12
Z13
[14
\15
]16
^17
_18
`19
a20
b21
c22
d23
e24
f25"
trackable_list_wrapper
æ
M0
N1
O2
P3
Q4
R5
S6
T7
U8
V9
W10
X11
Y12
Z13
[14
\15
]16
^17
_18
`19
a20
b21
c22
d23
e24
f25"
trackable_list_wrapper
 "
trackable_list_wrapper
Ê
gnon_trainable_variables

hlayers
imetrics
jlayer_regularization_losses
klayer_metrics
	variables
trainable_variables
	regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
õ
ltrace_0
mtrace_1
ntrace_2
otrace_32
0__inference_proto_ae_123321_layer_call_fn_551007
0__inference_proto_ae_123321_layer_call_fn_551486
0__inference_proto_ae_123321_layer_call_fn_551543
0__inference_proto_ae_123321_layer_call_fn_551240¿
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
 zltrace_0zmtrace_1zntrace_2zotrace_3
á
ptrace_0
qtrace_1
rtrace_2
strace_32ö
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551715
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551887
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551302
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551364¿
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
 zptrace_0zqtrace_1zrtrace_2zstrace_3
ÕBÒ
!__inference__wrapped_model_549767proto_enc1_input"
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
Û
titer

ubeta_1

vbeta_2
	wdecay
xlearning_rateMmNmOmPmQm Rm¡Sm¢Tm£Um¤Vm¥Wm¦Xm§Ym¨Zm©[mª\m«]m¬^m­_m®`m¯am°bm±cm²dm³em´fmµMv¶Nv·Ov¸Pv¹QvºRv»Sv¼Tv½Uv¾Vv¿WvÀXvÁYvÂZvÃ[vÄ\vÅ]vÆ^vÇ_vÈ`vÉavÊbvËcvÌdvÍevÎfvÏ"
	optimizer
,
yserving_default"
signature_map
D
#z_self_saveable_object_factories"
_tf_keras_input_layer

{	variables
|trainable_variables
}regularization_losses
~	keras_api
__call__
+&call_and_return_all_conditional_losses

Mkernel
Nbias
$_self_saveable_object_factories
!_jit_compiled_convolution_op"
_tf_keras_layer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

Okernel
Pbias
$_self_saveable_object_factories
!_jit_compiled_convolution_op"
_tf_keras_layer
<
M0
N1
O2
P3"
trackable_list_wrapper
<
M0
N1
O2
P3"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
é
trace_0
trace_1
trace_2
trace_32ö
+__inference_proto_enc1_layer_call_fn_549820
+__inference_proto_enc1_layer_call_fn_551900
+__inference_proto_enc1_layer_call_fn_551913
+__inference_proto_enc1_layer_call_fn_549893¿
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
 ztrace_0ztrace_1ztrace_2ztrace_3
Õ
trace_0
trace_1
trace_2
trace_32â
F__inference_proto_enc1_layer_call_and_return_conditional_losses_551931
F__inference_proto_enc1_layer_call_and_return_conditional_losses_551949
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549907
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549921¿
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
 ztrace_0ztrace_1ztrace_2ztrace_3
 "
trackable_dict_wrapper
E
$_self_saveable_object_factories"
_tf_keras_input_layer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

Qkernel
Rbias
$_self_saveable_object_factories
! _jit_compiled_convolution_op"
_tf_keras_layer

¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses

Skernel
Tbias
$§_self_saveable_object_factories
!¨_jit_compiled_convolution_op"
_tf_keras_layer
<
Q0
R1
S2
T3"
trackable_list_wrapper
<
Q0
R1
S2
T3"
trackable_list_wrapper
 "
trackable_list_wrapper
²
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
	variables
trainable_variables
regularization_losses
!__call__
*"&call_and_return_all_conditional_losses
&""call_and_return_conditional_losses"
_generic_user_object
é
®trace_0
¯trace_1
°trace_2
±trace_32ö
+__inference_proto_enc2_layer_call_fn_549974
+__inference_proto_enc2_layer_call_fn_551962
+__inference_proto_enc2_layer_call_fn_551975
+__inference_proto_enc2_layer_call_fn_550047¿
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
 z®trace_0z¯trace_1z°trace_2z±trace_3
Õ
²trace_0
³trace_1
´trace_2
µtrace_32â
F__inference_proto_enc2_layer_call_and_return_conditional_losses_551993
F__inference_proto_enc2_layer_call_and_return_conditional_losses_552011
F__inference_proto_enc2_layer_call_and_return_conditional_losses_550061
F__inference_proto_enc2_layer_call_and_return_conditional_losses_550075¿
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
 z²trace_0z³trace_1z´trace_2zµtrace_3
 "
trackable_dict_wrapper
E
$¶_self_saveable_object_factories"
_tf_keras_input_layer

·	variables
¸trainable_variables
¹regularization_losses
º	keras_api
»__call__
+¼&call_and_return_all_conditional_losses

Ukernel
Vbias
$½_self_saveable_object_factories
!¾_jit_compiled_convolution_op"
_tf_keras_layer

¿	variables
Àtrainable_variables
Áregularization_losses
Â	keras_api
Ã__call__
+Ä&call_and_return_all_conditional_losses

Wkernel
Xbias
$Å_self_saveable_object_factories
!Æ_jit_compiled_convolution_op"
_tf_keras_layer
<
U0
V1
W2
X3"
trackable_list_wrapper
<
U0
V1
W2
X3"
trackable_list_wrapper
 "
trackable_list_wrapper
²
Çnon_trainable_variables
Èlayers
Émetrics
 Êlayer_regularization_losses
Ëlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
é
Ìtrace_0
Ítrace_1
Îtrace_2
Ïtrace_32ö
+__inference_proto_enc3_layer_call_fn_550128
+__inference_proto_enc3_layer_call_fn_552024
+__inference_proto_enc3_layer_call_fn_552037
+__inference_proto_enc3_layer_call_fn_550201¿
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
 zÌtrace_0zÍtrace_1zÎtrace_2zÏtrace_3
Õ
Ðtrace_0
Ñtrace_1
Òtrace_2
Ótrace_32â
F__inference_proto_enc3_layer_call_and_return_conditional_losses_552055
F__inference_proto_enc3_layer_call_and_return_conditional_losses_552073
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550215
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550229¿
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
 zÐtrace_0zÑtrace_1zÒtrace_2zÓtrace_3
 "
trackable_dict_wrapper
E
$Ô_self_saveable_object_factories"
_tf_keras_input_layer

Õ	variables
Ötrainable_variables
×regularization_losses
Ø	keras_api
Ù__call__
+Ú&call_and_return_all_conditional_losses

Ykernel
Zbias
$Û_self_saveable_object_factories
!Ü_jit_compiled_convolution_op"
_tf_keras_layer

Ý	variables
Þtrainable_variables
ßregularization_losses
à	keras_api
á__call__
+â&call_and_return_all_conditional_losses

[kernel
\bias
$ã_self_saveable_object_factories
!ä_jit_compiled_convolution_op"
_tf_keras_layer
<
Y0
Z1
[2
\3"
trackable_list_wrapper
<
Y0
Z1
[2
\3"
trackable_list_wrapper
 "
trackable_list_wrapper
²
ånon_trainable_variables
ælayers
çmetrics
 èlayer_regularization_losses
élayer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
é
êtrace_0
ëtrace_1
ìtrace_2
ítrace_32ö
+__inference_proto_dec3_layer_call_fn_550348
+__inference_proto_dec3_layer_call_fn_552086
+__inference_proto_dec3_layer_call_fn_552099
+__inference_proto_dec3_layer_call_fn_550401¿
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
 zêtrace_0zëtrace_1zìtrace_2zítrace_3
Õ
îtrace_0
ïtrace_1
ðtrace_2
ñtrace_32â
F__inference_proto_dec3_layer_call_and_return_conditional_losses_552143
F__inference_proto_dec3_layer_call_and_return_conditional_losses_552187
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550415
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550429¿
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
 zîtrace_0zïtrace_1zðtrace_2zñtrace_3
 "
trackable_dict_wrapper
E
$ò_self_saveable_object_factories"
_tf_keras_input_layer

ó	variables
ôtrainable_variables
õregularization_losses
ö	keras_api
÷__call__
+ø&call_and_return_all_conditional_losses

]kernel
^bias
$ù_self_saveable_object_factories
!ú_jit_compiled_convolution_op"
_tf_keras_layer

û	variables
ütrainable_variables
ýregularization_losses
þ	keras_api
ÿ__call__
+&call_and_return_all_conditional_losses

_kernel
`bias
$_self_saveable_object_factories
!_jit_compiled_convolution_op"
_tf_keras_layer
<
]0
^1
_2
`3"
trackable_list_wrapper
<
]0
^1
_2
`3"
trackable_list_wrapper
 "
trackable_list_wrapper
²
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
é
trace_0
trace_1
trace_2
trace_32ö
+__inference_proto_dec2_layer_call_fn_550548
+__inference_proto_dec2_layer_call_fn_552200
+__inference_proto_dec2_layer_call_fn_552213
+__inference_proto_dec2_layer_call_fn_550601¿
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
 ztrace_0ztrace_1ztrace_2ztrace_3
Õ
trace_0
trace_1
trace_2
trace_32â
F__inference_proto_dec2_layer_call_and_return_conditional_losses_552257
F__inference_proto_dec2_layer_call_and_return_conditional_losses_552301
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550615
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550629¿
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
 ztrace_0ztrace_1ztrace_2ztrace_3
 "
trackable_dict_wrapper
E
$_self_saveable_object_factories"
_tf_keras_input_layer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

akernel
bbias
$_self_saveable_object_factories
!_jit_compiled_convolution_op"
_tf_keras_layer

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses

ckernel
dbias
$_self_saveable_object_factories
! _jit_compiled_convolution_op"
_tf_keras_layer

¡	variables
¢trainable_variables
£regularization_losses
¤	keras_api
¥__call__
+¦&call_and_return_all_conditional_losses

ekernel
fbias
$§_self_saveable_object_factories
!¨_jit_compiled_convolution_op"
_tf_keras_layer
J
a0
b1
c2
d3
e4
f5"
trackable_list_wrapper
J
a0
b1
c2
d3
e4
f5"
trackable_list_wrapper
 "
trackable_list_wrapper
²
©non_trainable_variables
ªlayers
«metrics
 ¬layer_regularization_losses
­layer_metrics
F	variables
Gtrainable_variables
Hregularization_losses
J__call__
*K&call_and_return_all_conditional_losses
&K"call_and_return_conditional_losses"
_generic_user_object
é
®trace_0
¯trace_1
°trace_2
±trace_32ö
+__inference_proto_dec1_layer_call_fn_550768
+__inference_proto_dec1_layer_call_fn_552318
+__inference_proto_dec1_layer_call_fn_552335
+__inference_proto_dec1_layer_call_fn_550848¿
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
 z®trace_0z¯trace_1z°trace_2z±trace_3
Õ
²trace_0
³trace_1
´trace_2
µtrace_32â
F__inference_proto_dec1_layer_call_and_return_conditional_losses_552385
F__inference_proto_dec1_layer_call_and_return_conditional_losses_552435
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550867
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550886¿
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
 z²trace_0z³trace_1z´trace_2zµtrace_3
 "
trackable_dict_wrapper
':%	2conv2d/kernel
:	2conv2d/bias
):'		2conv2d_1/kernel
:	2conv2d_1/bias
):'	2conv2d_3/kernel
:2conv2d_3/bias
):'2conv2d_4/kernel
:2conv2d_4/bias
):'2conv2d_5/kernel
:2conv2d_5/bias
):'2conv2d_6/kernel
:2conv2d_6/bias
3:12conv2d_transpose_4/kernel
%:#2conv2d_transpose_4/bias
3:12conv2d_transpose_5/kernel
%:#2conv2d_transpose_5/bias
3:1	2conv2d_transpose_2/kernel
%:#	2conv2d_transpose_2/bias
3:1		2conv2d_transpose_3/kernel
%:#	2conv2d_transpose_3/bias
1:/		2conv2d_transpose/kernel
#:!	2conv2d_transpose/bias
3:1		2conv2d_transpose_1/kernel
%:#	2conv2d_transpose_1/bias
):'	2conv2d_2/kernel
:2conv2d_2/bias
 "
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
0
¶0
·1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
0__inference_proto_ae_123321_layer_call_fn_551007proto_enc1_input"¿
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
Bþ
0__inference_proto_ae_123321_layer_call_fn_551486inputs"¿
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
Bþ
0__inference_proto_ae_123321_layer_call_fn_551543inputs"¿
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
B
0__inference_proto_ae_123321_layer_call_fn_551240proto_enc1_input"¿
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
B
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551715inputs"¿
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
B
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551887inputs"¿
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
¦B£
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551302proto_enc1_input"¿
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
¦B£
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551364proto_enc1_input"¿
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
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÔBÑ
$__inference_signature_wrapper_551429proto_enc1_input"
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
trackable_dict_wrapper
.
M0
N1"
trackable_list_wrapper
.
M0
N1"
trackable_list_wrapper
 "
trackable_list_wrapper
´
¸non_trainable_variables
¹layers
ºmetrics
 »layer_regularization_losses
¼layer_metrics
{	variables
|trainable_variables
}regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
í
½trace_02Î
'__inference_conv2d_layer_call_fn_552444¢
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
 z½trace_0

¾trace_02é
B__inference_conv2d_layer_call_and_return_conditional_losses_552455¢
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
 z¾trace_0
 "
trackable_dict_wrapper
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ï
Ätrace_02Ð
)__inference_conv2d_1_layer_call_fn_552464¢
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
 zÄtrace_0

Åtrace_02ë
D__inference_conv2d_1_layer_call_and_return_conditional_losses_552475¢
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
 zÅtrace_0
 "
trackable_dict_wrapper
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ýBú
+__inference_proto_enc1_layer_call_fn_549820input_1"¿
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
üBù
+__inference_proto_enc1_layer_call_fn_551900inputs"¿
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
üBù
+__inference_proto_enc1_layer_call_fn_551913inputs"¿
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
ýBú
+__inference_proto_enc1_layer_call_fn_549893input_1"¿
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
B
F__inference_proto_enc1_layer_call_and_return_conditional_losses_551931inputs"¿
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
B
F__inference_proto_enc1_layer_call_and_return_conditional_losses_551949inputs"¿
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
B
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549907input_1"¿
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
B
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549921input_1"¿
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
 "
trackable_dict_wrapper
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ï
Ëtrace_02Ð
)__inference_conv2d_3_layer_call_fn_552484¢
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
 zËtrace_0

Ìtrace_02ë
D__inference_conv2d_3_layer_call_and_return_conditional_losses_552495¢
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
 zÌtrace_0
 "
trackable_dict_wrapper
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
.
S0
T1"
trackable_list_wrapper
.
S0
T1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
ï
Òtrace_02Ð
)__inference_conv2d_4_layer_call_fn_552504¢
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
 zÒtrace_0

Ótrace_02ë
D__inference_conv2d_4_layer_call_and_return_conditional_losses_552515¢
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
 zÓtrace_0
 "
trackable_dict_wrapper
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ýBú
+__inference_proto_enc2_layer_call_fn_549974input_3"¿
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
üBù
+__inference_proto_enc2_layer_call_fn_551962inputs"¿
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
üBù
+__inference_proto_enc2_layer_call_fn_551975inputs"¿
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
ýBú
+__inference_proto_enc2_layer_call_fn_550047input_3"¿
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
B
F__inference_proto_enc2_layer_call_and_return_conditional_losses_551993inputs"¿
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
B
F__inference_proto_enc2_layer_call_and_return_conditional_losses_552011inputs"¿
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
B
F__inference_proto_enc2_layer_call_and_return_conditional_losses_550061input_3"¿
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
B
F__inference_proto_enc2_layer_call_and_return_conditional_losses_550075input_3"¿
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
 "
trackable_dict_wrapper
.
U0
V1"
trackable_list_wrapper
.
U0
V1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
·	variables
¸trainable_variables
¹regularization_losses
»__call__
+¼&call_and_return_all_conditional_losses
'¼"call_and_return_conditional_losses"
_generic_user_object
ï
Ùtrace_02Ð
)__inference_conv2d_5_layer_call_fn_552524¢
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
 zÙtrace_0

Útrace_02ë
D__inference_conv2d_5_layer_call_and_return_conditional_losses_552535¢
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
 zÚtrace_0
 "
trackable_dict_wrapper
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
.
W0
X1"
trackable_list_wrapper
.
W0
X1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ûnon_trainable_variables
Ülayers
Ýmetrics
 Þlayer_regularization_losses
ßlayer_metrics
¿	variables
Àtrainable_variables
Áregularization_losses
Ã__call__
+Ä&call_and_return_all_conditional_losses
'Ä"call_and_return_conditional_losses"
_generic_user_object
ï
àtrace_02Ð
)__inference_conv2d_6_layer_call_fn_552544¢
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
 zàtrace_0

átrace_02ë
D__inference_conv2d_6_layer_call_and_return_conditional_losses_552555¢
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
 zátrace_0
 "
trackable_dict_wrapper
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 "
trackable_list_wrapper
5
$0
%1
&2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ýBú
+__inference_proto_enc3_layer_call_fn_550128input_5"¿
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
üBù
+__inference_proto_enc3_layer_call_fn_552024inputs"¿
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
üBù
+__inference_proto_enc3_layer_call_fn_552037inputs"¿
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
ýBú
+__inference_proto_enc3_layer_call_fn_550201input_5"¿
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
B
F__inference_proto_enc3_layer_call_and_return_conditional_losses_552055inputs"¿
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
B
F__inference_proto_enc3_layer_call_and_return_conditional_losses_552073inputs"¿
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
B
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550215input_5"¿
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
B
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550229input_5"¿
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
 "
trackable_dict_wrapper
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ânon_trainable_variables
ãlayers
ämetrics
 ålayer_regularization_losses
ælayer_metrics
Õ	variables
Ötrainable_variables
×regularization_losses
Ù__call__
+Ú&call_and_return_all_conditional_losses
'Ú"call_and_return_conditional_losses"
_generic_user_object
ù
çtrace_02Ú
3__inference_conv2d_transpose_4_layer_call_fn_552564¢
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
 zçtrace_0

ètrace_02õ
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_552598¢
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
 zètrace_0
 "
trackable_dict_wrapper
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
énon_trainable_variables
êlayers
ëmetrics
 ìlayer_regularization_losses
ílayer_metrics
Ý	variables
Þtrainable_variables
ßregularization_losses
á__call__
+â&call_and_return_all_conditional_losses
'â"call_and_return_conditional_losses"
_generic_user_object
ù
îtrace_02Ú
3__inference_conv2d_transpose_5_layer_call_fn_552607¢
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
 zîtrace_0

ïtrace_02õ
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_552641¢
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
 zïtrace_0
 "
trackable_dict_wrapper
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 "
trackable_list_wrapper
5
.0
/1
02"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ýBú
+__inference_proto_dec3_layer_call_fn_550348input_6"¿
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
üBù
+__inference_proto_dec3_layer_call_fn_552086inputs"¿
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
üBù
+__inference_proto_dec3_layer_call_fn_552099inputs"¿
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
ýBú
+__inference_proto_dec3_layer_call_fn_550401input_6"¿
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
B
F__inference_proto_dec3_layer_call_and_return_conditional_losses_552143inputs"¿
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
B
F__inference_proto_dec3_layer_call_and_return_conditional_losses_552187inputs"¿
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
B
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550415input_6"¿
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
B
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550429input_6"¿
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
 "
trackable_dict_wrapper
.
]0
^1"
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
ðnon_trainable_variables
ñlayers
òmetrics
 ólayer_regularization_losses
ôlayer_metrics
ó	variables
ôtrainable_variables
õregularization_losses
÷__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses"
_generic_user_object
ù
õtrace_02Ú
3__inference_conv2d_transpose_2_layer_call_fn_552650¢
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
 zõtrace_0

ötrace_02õ
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_552684¢
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
 zötrace_0
 "
trackable_dict_wrapper
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
.
_0
`1"
trackable_list_wrapper
.
_0
`1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
÷non_trainable_variables
ølayers
ùmetrics
 úlayer_regularization_losses
ûlayer_metrics
û	variables
ütrainable_variables
ýregularization_losses
ÿ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ù
ütrace_02Ú
3__inference_conv2d_transpose_3_layer_call_fn_552693¢
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
 zütrace_0

ýtrace_02õ
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_552727¢
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
 zýtrace_0
 "
trackable_dict_wrapper
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 "
trackable_list_wrapper
5
80
91
:2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ýBú
+__inference_proto_dec2_layer_call_fn_550548input_4"¿
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
üBù
+__inference_proto_dec2_layer_call_fn_552200inputs"¿
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
üBù
+__inference_proto_dec2_layer_call_fn_552213inputs"¿
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
ýBú
+__inference_proto_dec2_layer_call_fn_550601input_4"¿
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
B
F__inference_proto_dec2_layer_call_and_return_conditional_losses_552257inputs"¿
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
B
F__inference_proto_dec2_layer_call_and_return_conditional_losses_552301inputs"¿
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
B
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550615input_4"¿
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
B
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550629input_4"¿
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
 "
trackable_dict_wrapper
.
a0
b1"
trackable_list_wrapper
.
a0
b1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
þnon_trainable_variables
ÿlayers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
÷
trace_02Ø
1__inference_conv2d_transpose_layer_call_fn_552736¢
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
 ztrace_0

trace_02ó
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_552770¢
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
 ztrace_0
 "
trackable_dict_wrapper
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
.
c0
d1"
trackable_list_wrapper
.
c0
d1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ù
trace_02Ú
3__inference_conv2d_transpose_1_layer_call_fn_552779¢
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
 ztrace_0

trace_02õ
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_552813¢
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
 ztrace_0
 "
trackable_dict_wrapper
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
¡	variables
¢trainable_variables
£regularization_losses
¥__call__
+¦&call_and_return_all_conditional_losses
'¦"call_and_return_conditional_losses"
_generic_user_object
ï
trace_02Ð
)__inference_conv2d_2_layer_call_fn_552822¢
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
 ztrace_0

trace_02ë
D__inference_conv2d_2_layer_call_and_return_conditional_losses_552832¢
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
 ztrace_0
 "
trackable_dict_wrapper
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
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
 "
trackable_list_wrapper
<
B0
C1
D2
E3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ýBú
+__inference_proto_dec1_layer_call_fn_550768input_2"¿
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
üBù
+__inference_proto_dec1_layer_call_fn_552318inputs"¿
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
üBù
+__inference_proto_dec1_layer_call_fn_552335inputs"¿
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
ýBú
+__inference_proto_dec1_layer_call_fn_550848input_2"¿
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
B
F__inference_proto_dec1_layer_call_and_return_conditional_losses_552385inputs"¿
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
B
F__inference_proto_dec1_layer_call_and_return_conditional_losses_552435inputs"¿
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
B
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550867input_2"¿
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
B
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550886input_2"¿
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
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
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
ÛBØ
'__inference_conv2d_layer_call_fn_552444inputs"¢
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
öBó
B__inference_conv2d_layer_call_and_return_conditional_losses_552455inputs"¢
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
ÝBÚ
)__inference_conv2d_1_layer_call_fn_552464inputs"¢
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
øBõ
D__inference_conv2d_1_layer_call_and_return_conditional_losses_552475inputs"¢
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
ÝBÚ
)__inference_conv2d_3_layer_call_fn_552484inputs"¢
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
øBõ
D__inference_conv2d_3_layer_call_and_return_conditional_losses_552495inputs"¢
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
ÝBÚ
)__inference_conv2d_4_layer_call_fn_552504inputs"¢
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
øBõ
D__inference_conv2d_4_layer_call_and_return_conditional_losses_552515inputs"¢
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
ÝBÚ
)__inference_conv2d_5_layer_call_fn_552524inputs"¢
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
øBõ
D__inference_conv2d_5_layer_call_and_return_conditional_losses_552535inputs"¢
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
ÝBÚ
)__inference_conv2d_6_layer_call_fn_552544inputs"¢
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
øBõ
D__inference_conv2d_6_layer_call_and_return_conditional_losses_552555inputs"¢
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
çBä
3__inference_conv2d_transpose_4_layer_call_fn_552564inputs"¢
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
Bÿ
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_552598inputs"¢
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
çBä
3__inference_conv2d_transpose_5_layer_call_fn_552607inputs"¢
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
Bÿ
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_552641inputs"¢
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
çBä
3__inference_conv2d_transpose_2_layer_call_fn_552650inputs"¢
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
Bÿ
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_552684inputs"¢
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
çBä
3__inference_conv2d_transpose_3_layer_call_fn_552693inputs"¢
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
Bÿ
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_552727inputs"¢
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
åBâ
1__inference_conv2d_transpose_layer_call_fn_552736inputs"¢
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
Bý
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_552770inputs"¢
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
çBä
3__inference_conv2d_transpose_1_layer_call_fn_552779inputs"¢
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
Bÿ
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_552813inputs"¢
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
ÝBÚ
)__inference_conv2d_2_layer_call_fn_552822inputs"¢
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
øBõ
D__inference_conv2d_2_layer_call_and_return_conditional_losses_552832inputs"¢
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
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
,:*	2Adam/conv2d/kernel/m
:	2Adam/conv2d/bias/m
.:,		2Adam/conv2d_1/kernel/m
 :	2Adam/conv2d_1/bias/m
.:,	2Adam/conv2d_3/kernel/m
 :2Adam/conv2d_3/bias/m
.:,2Adam/conv2d_4/kernel/m
 :2Adam/conv2d_4/bias/m
.:,2Adam/conv2d_5/kernel/m
 :2Adam/conv2d_5/bias/m
.:,2Adam/conv2d_6/kernel/m
 :2Adam/conv2d_6/bias/m
8:62 Adam/conv2d_transpose_4/kernel/m
*:(2Adam/conv2d_transpose_4/bias/m
8:62 Adam/conv2d_transpose_5/kernel/m
*:(2Adam/conv2d_transpose_5/bias/m
8:6	2 Adam/conv2d_transpose_2/kernel/m
*:(	2Adam/conv2d_transpose_2/bias/m
8:6		2 Adam/conv2d_transpose_3/kernel/m
*:(	2Adam/conv2d_transpose_3/bias/m
6:4		2Adam/conv2d_transpose/kernel/m
(:&	2Adam/conv2d_transpose/bias/m
8:6		2 Adam/conv2d_transpose_1/kernel/m
*:(	2Adam/conv2d_transpose_1/bias/m
.:,	2Adam/conv2d_2/kernel/m
 :2Adam/conv2d_2/bias/m
,:*	2Adam/conv2d/kernel/v
:	2Adam/conv2d/bias/v
.:,		2Adam/conv2d_1/kernel/v
 :	2Adam/conv2d_1/bias/v
.:,	2Adam/conv2d_3/kernel/v
 :2Adam/conv2d_3/bias/v
.:,2Adam/conv2d_4/kernel/v
 :2Adam/conv2d_4/bias/v
.:,2Adam/conv2d_5/kernel/v
 :2Adam/conv2d_5/bias/v
.:,2Adam/conv2d_6/kernel/v
 :2Adam/conv2d_6/bias/v
8:62 Adam/conv2d_transpose_4/kernel/v
*:(2Adam/conv2d_transpose_4/bias/v
8:62 Adam/conv2d_transpose_5/kernel/v
*:(2Adam/conv2d_transpose_5/bias/v
8:6	2 Adam/conv2d_transpose_2/kernel/v
*:(	2Adam/conv2d_transpose_2/bias/v
8:6		2 Adam/conv2d_transpose_3/kernel/v
*:(	2Adam/conv2d_transpose_3/bias/v
6:4		2Adam/conv2d_transpose/kernel/v
(:&	2Adam/conv2d_transpose/bias/v
8:6		2 Adam/conv2d_transpose_1/kernel/v
*:(	2Adam/conv2d_transpose_1/bias/v
.:,	2Adam/conv2d_2/kernel/v
 :2Adam/conv2d_2/bias/vÆ
!__inference__wrapped_model_549767 MNOPQRSTUVWXYZ[\]^_`abcdefA¢>
7¢4
2/
proto_enc1_inputÿÿÿÿÿÿÿÿÿdd
ª "?ª<
:

proto_dec1,)

proto_dec1ÿÿÿÿÿÿÿÿÿdd´
D__inference_conv2d_1_layer_call_and_return_conditional_losses_552475lOP7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22	
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22	
 
)__inference_conv2d_1_layer_call_fn_552464_OP7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22	
ª " ÿÿÿÿÿÿÿÿÿ22	´
D__inference_conv2d_2_layer_call_and_return_conditional_losses_552832lef7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿdd	
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿdd
 
)__inference_conv2d_2_layer_call_fn_552822_ef7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿdd	
ª " ÿÿÿÿÿÿÿÿÿdd´
D__inference_conv2d_3_layer_call_and_return_conditional_losses_552495lQR7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22	
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 
)__inference_conv2d_3_layer_call_fn_552484_QR7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22	
ª " ÿÿÿÿÿÿÿÿÿ22´
D__inference_conv2d_4_layer_call_and_return_conditional_losses_552515lST7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 
)__inference_conv2d_4_layer_call_fn_552504_ST7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22
ª " ÿÿÿÿÿÿÿÿÿ22´
D__inference_conv2d_5_layer_call_and_return_conditional_losses_552535lUV7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 
)__inference_conv2d_5_layer_call_fn_552524_UV7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22
ª " ÿÿÿÿÿÿÿÿÿ22´
D__inference_conv2d_6_layer_call_and_return_conditional_losses_552555lWX7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 
)__inference_conv2d_6_layer_call_fn_552544_WX7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ22
ª " ÿÿÿÿÿÿÿÿÿ22²
B__inference_conv2d_layer_call_and_return_conditional_losses_552455lMN7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿdd
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22	
 
'__inference_conv2d_layer_call_fn_552444_MN7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿdd
ª " ÿÿÿÿÿÿÿÿÿ22	ã
N__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_552813cdI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
 »
3__inference_conv2d_transpose_1_layer_call_fn_552779cdI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	ã
N__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_552684]^I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
 »
3__inference_conv2d_transpose_2_layer_call_fn_552650]^I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	ã
N__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_552727_`I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
 »
3__inference_conv2d_transpose_3_layer_call_fn_552693_`I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	ã
N__inference_conv2d_transpose_4_layer_call_and_return_conditional_losses_552598YZI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 »
3__inference_conv2d_transpose_4_layer_call_fn_552564YZI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿã
N__inference_conv2d_transpose_5_layer_call_and_return_conditional_losses_552641[\I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 »
3__inference_conv2d_transpose_5_layer_call_fn_552607[\I¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿá
L__inference_conv2d_transpose_layer_call_and_return_conditional_losses_552770abI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
 ¹
1__inference_conv2d_transpose_layer_call_fn_552736abI¢F
?¢<
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ	æ
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551302MNOPQRSTUVWXYZ[\]^_`abcdefI¢F
?¢<
2/
proto_enc1_inputÿÿÿÿÿÿÿÿÿdd
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿdd
 æ
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551364MNOPQRSTUVWXYZ[\]^_`abcdefI¢F
?¢<
2/
proto_enc1_inputÿÿÿÿÿÿÿÿÿdd
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿdd
 Ü
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551715MNOPQRSTUVWXYZ[\]^_`abcdef?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿdd
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿdd
 Ü
K__inference_proto_ae_123321_layer_call_and_return_conditional_losses_551887MNOPQRSTUVWXYZ[\]^_`abcdef?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿdd
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿdd
 ¾
0__inference_proto_ae_123321_layer_call_fn_551007MNOPQRSTUVWXYZ[\]^_`abcdefI¢F
?¢<
2/
proto_enc1_inputÿÿÿÿÿÿÿÿÿdd
p 

 
ª " ÿÿÿÿÿÿÿÿÿdd¾
0__inference_proto_ae_123321_layer_call_fn_551240MNOPQRSTUVWXYZ[\]^_`abcdefI¢F
?¢<
2/
proto_enc1_inputÿÿÿÿÿÿÿÿÿdd
p

 
ª " ÿÿÿÿÿÿÿÿÿdd³
0__inference_proto_ae_123321_layer_call_fn_551486MNOPQRSTUVWXYZ[\]^_`abcdef?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿdd
p 

 
ª " ÿÿÿÿÿÿÿÿÿdd³
0__inference_proto_ae_123321_layer_call_fn_551543MNOPQRSTUVWXYZ[\]^_`abcdef?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿdd
p

 
ª " ÿÿÿÿÿÿÿÿÿddÃ
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550867yabcdef@¢=
6¢3
)&
input_2ÿÿÿÿÿÿÿÿÿ22	
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿdd
 Ã
F__inference_proto_dec1_layer_call_and_return_conditional_losses_550886yabcdef@¢=
6¢3
)&
input_2ÿÿÿÿÿÿÿÿÿ22	
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿdd
 Â
F__inference_proto_dec1_layer_call_and_return_conditional_losses_552385xabcdef?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22	
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿdd
 Â
F__inference_proto_dec1_layer_call_and_return_conditional_losses_552435xabcdef?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22	
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿdd
 
+__inference_proto_dec1_layer_call_fn_550768labcdef@¢=
6¢3
)&
input_2ÿÿÿÿÿÿÿÿÿ22	
p 

 
ª " ÿÿÿÿÿÿÿÿÿdd
+__inference_proto_dec1_layer_call_fn_550848labcdef@¢=
6¢3
)&
input_2ÿÿÿÿÿÿÿÿÿ22	
p

 
ª " ÿÿÿÿÿÿÿÿÿdd
+__inference_proto_dec1_layer_call_fn_552318kabcdef?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22	
p 

 
ª " ÿÿÿÿÿÿÿÿÿdd
+__inference_proto_dec1_layer_call_fn_552335kabcdef?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22	
p

 
ª " ÿÿÿÿÿÿÿÿÿddÁ
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550615w]^_`@¢=
6¢3
)&
input_4ÿÿÿÿÿÿÿÿÿ22
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22	
 Á
F__inference_proto_dec2_layer_call_and_return_conditional_losses_550629w]^_`@¢=
6¢3
)&
input_4ÿÿÿÿÿÿÿÿÿ22
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22	
 À
F__inference_proto_dec2_layer_call_and_return_conditional_losses_552257v]^_`?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22	
 À
F__inference_proto_dec2_layer_call_and_return_conditional_losses_552301v]^_`?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22	
 
+__inference_proto_dec2_layer_call_fn_550548j]^_`@¢=
6¢3
)&
input_4ÿÿÿÿÿÿÿÿÿ22
p 

 
ª " ÿÿÿÿÿÿÿÿÿ22	
+__inference_proto_dec2_layer_call_fn_550601j]^_`@¢=
6¢3
)&
input_4ÿÿÿÿÿÿÿÿÿ22
p

 
ª " ÿÿÿÿÿÿÿÿÿ22	
+__inference_proto_dec2_layer_call_fn_552200i]^_`?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p 

 
ª " ÿÿÿÿÿÿÿÿÿ22	
+__inference_proto_dec2_layer_call_fn_552213i]^_`?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p

 
ª " ÿÿÿÿÿÿÿÿÿ22	Á
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550415wYZ[\@¢=
6¢3
)&
input_6ÿÿÿÿÿÿÿÿÿ22
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 Á
F__inference_proto_dec3_layer_call_and_return_conditional_losses_550429wYZ[\@¢=
6¢3
)&
input_6ÿÿÿÿÿÿÿÿÿ22
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 À
F__inference_proto_dec3_layer_call_and_return_conditional_losses_552143vYZ[\?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 À
F__inference_proto_dec3_layer_call_and_return_conditional_losses_552187vYZ[\?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 
+__inference_proto_dec3_layer_call_fn_550348jYZ[\@¢=
6¢3
)&
input_6ÿÿÿÿÿÿÿÿÿ22
p 

 
ª " ÿÿÿÿÿÿÿÿÿ22
+__inference_proto_dec3_layer_call_fn_550401jYZ[\@¢=
6¢3
)&
input_6ÿÿÿÿÿÿÿÿÿ22
p

 
ª " ÿÿÿÿÿÿÿÿÿ22
+__inference_proto_dec3_layer_call_fn_552086iYZ[\?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p 

 
ª " ÿÿÿÿÿÿÿÿÿ22
+__inference_proto_dec3_layer_call_fn_552099iYZ[\?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p

 
ª " ÿÿÿÿÿÿÿÿÿ22Á
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549907wMNOP@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿdd
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22	
 Á
F__inference_proto_enc1_layer_call_and_return_conditional_losses_549921wMNOP@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿdd
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22	
 À
F__inference_proto_enc1_layer_call_and_return_conditional_losses_551931vMNOP?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿdd
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22	
 À
F__inference_proto_enc1_layer_call_and_return_conditional_losses_551949vMNOP?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿdd
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22	
 
+__inference_proto_enc1_layer_call_fn_549820jMNOP@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿdd
p 

 
ª " ÿÿÿÿÿÿÿÿÿ22	
+__inference_proto_enc1_layer_call_fn_549893jMNOP@¢=
6¢3
)&
input_1ÿÿÿÿÿÿÿÿÿdd
p

 
ª " ÿÿÿÿÿÿÿÿÿ22	
+__inference_proto_enc1_layer_call_fn_551900iMNOP?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿdd
p 

 
ª " ÿÿÿÿÿÿÿÿÿ22	
+__inference_proto_enc1_layer_call_fn_551913iMNOP?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿdd
p

 
ª " ÿÿÿÿÿÿÿÿÿ22	Á
F__inference_proto_enc2_layer_call_and_return_conditional_losses_550061wQRST@¢=
6¢3
)&
input_3ÿÿÿÿÿÿÿÿÿ22	
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 Á
F__inference_proto_enc2_layer_call_and_return_conditional_losses_550075wQRST@¢=
6¢3
)&
input_3ÿÿÿÿÿÿÿÿÿ22	
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 À
F__inference_proto_enc2_layer_call_and_return_conditional_losses_551993vQRST?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22	
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 À
F__inference_proto_enc2_layer_call_and_return_conditional_losses_552011vQRST?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22	
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 
+__inference_proto_enc2_layer_call_fn_549974jQRST@¢=
6¢3
)&
input_3ÿÿÿÿÿÿÿÿÿ22	
p 

 
ª " ÿÿÿÿÿÿÿÿÿ22
+__inference_proto_enc2_layer_call_fn_550047jQRST@¢=
6¢3
)&
input_3ÿÿÿÿÿÿÿÿÿ22	
p

 
ª " ÿÿÿÿÿÿÿÿÿ22
+__inference_proto_enc2_layer_call_fn_551962iQRST?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22	
p 

 
ª " ÿÿÿÿÿÿÿÿÿ22
+__inference_proto_enc2_layer_call_fn_551975iQRST?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22	
p

 
ª " ÿÿÿÿÿÿÿÿÿ22Á
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550215wUVWX@¢=
6¢3
)&
input_5ÿÿÿÿÿÿÿÿÿ22
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 Á
F__inference_proto_enc3_layer_call_and_return_conditional_losses_550229wUVWX@¢=
6¢3
)&
input_5ÿÿÿÿÿÿÿÿÿ22
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 À
F__inference_proto_enc3_layer_call_and_return_conditional_losses_552055vUVWX?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p 

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 À
F__inference_proto_enc3_layer_call_and_return_conditional_losses_552073vUVWX?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p

 
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ22
 
+__inference_proto_enc3_layer_call_fn_550128jUVWX@¢=
6¢3
)&
input_5ÿÿÿÿÿÿÿÿÿ22
p 

 
ª " ÿÿÿÿÿÿÿÿÿ22
+__inference_proto_enc3_layer_call_fn_550201jUVWX@¢=
6¢3
)&
input_5ÿÿÿÿÿÿÿÿÿ22
p

 
ª " ÿÿÿÿÿÿÿÿÿ22
+__inference_proto_enc3_layer_call_fn_552024iUVWX?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p 

 
ª " ÿÿÿÿÿÿÿÿÿ22
+__inference_proto_enc3_layer_call_fn_552037iUVWX?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ22
p

 
ª " ÿÿÿÿÿÿÿÿÿ22Ý
$__inference_signature_wrapper_551429´MNOPQRSTUVWXYZ[\]^_`abcdefU¢R
¢ 
KªH
F
proto_enc1_input2/
proto_enc1_inputÿÿÿÿÿÿÿÿÿdd"?ª<
:

proto_dec1,)

proto_dec1ÿÿÿÿÿÿÿÿÿdd