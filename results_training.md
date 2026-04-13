ubuntu@ip-172-31-40-51:~$ ~source tf_gpu_env/bin/activate
~source: command not found
ubuntu@ip-172-31-40-51:~$ source ~/tf_gpu_env/bin/activate
(tf_gpu_env) ubuntu@ip-172-31-40-51:~$ python fixed_CNN_training.py
python: can't open file '/home/ubuntu/fixed_CNN_training.py': [Errno 2] No such file or directory
(tf_gpu_env) ubuntu@ip-172-31-40-51:~$ python scripts/fixed_CNN_training.py
2026-04-10 14:14:08.106451: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:479] Unable to register cuFFT factory: Attempting to register factory for plug
in cuFFT when one has already been registered
2026-04-10 14:14:08.362021: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:10575] Unable to register cuDNN factory: Attempting to register factory for pl
ugin cuDNN when one has already been registered
2026-04-10 14:14:08.363921: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1442] Unable to register cuBLAS factory: Attempting to register factory for p
lugin cuBLAS when one has already been registered
2026-04-10 14:14:08.808716: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performanc
e-critical operations.
To enable the following instructions: AVX2 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
2026-04-10 14:14:10.530312: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT
Casting to class labels:  32%|███████████████████████████▌                                                          | 32000/100000 [00:06<00:13, 5151.52 examples/s]
Casting to class labels:  37%|███████████████████████████████▊                                                      | 37000/100000 [00:07<00:12, 5056.81 examples/s]
Casting to class labels:  38%|████████████████████████████████▋                                                     | 38000/100000 [00:07<00:12, 4973.83 examples/s]
Casting to class labels:  40%|██████████████████████████████████▍                                                   | 40000/100000 [00:08<00:12, 4682.33 examples/s]
Casting to class labels: 100%|█████████████████████████████████████████████████████████████████████████████████████| 100000/100000 [00:48<00:00, 2073.47 examples/s]
Saving the dataset (4/4 shards): 100%|███████████████████████████████████████████████████████████████████████████████| 20000/20000 [00:14<00:00, 1364.49 examples/s]
Test-set sparat
2026-04-10 14:15:32.615749: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1928] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 20833 MB memor
y:  -> device: 0, name: NVIDIA L4, pci bus id: 0000:31:00.0, compute capability: 8.9
Epoch 1/30
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
I0000 00:00:1775830543.288866    3978 service.cc:145] XLA service 0x70b5bc002400 initialized for platform CUDA (this does not guarantee that XLA will be used). Devi
ces:
I0000 00:00:1775830543.288923    3978 service.cc:153]   StreamExecutor device (0): NVIDIA L4, Compute Capability 8.9
2026-04-10 14:15:43.457639: I tensorflow/compiler/mlir/tensorflow/utils/dump_mlir_util.cc:268] disabling MLIR crash reproducer, set env var `MLIR_CRASH_REPRODUCER_D
IRECTORY` to enable.
2026-04-10 14:15:44.392565: I external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:465] Loaded cuDNN version 8907
I0000 00:00:1775830555.721751    3978 device_compiler.h:188] Compiled cluster using XLA!  This line is logged at most once for the lifetime of the process.
   2000/Unknown 255s 117ms/step - accuracy: 0.5633 - loss: 0.72562026-04-10 14:19:50.284277: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous i
s aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 14:19:50.284489: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
/home/ubuntu/tf_gpu_env/lib/python3.10/site-packages/keras/src/trainers/epoch_iterator.py:164: UserWarning: Your input ran out of data; interrupting training. Make
sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your datase
t.
  self._interrupted_warning()
2026-04-10 14:20:52.060443: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 14:20:52.060551: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 317s 148ms/step - accuracy: 0.6141 - loss: 0.6682 - val_accuracy: 0.7232 - val_loss: 0.5618
Epoch 2/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.6911 - loss: 0.58672026-04-10 14:24:56.207363: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 14:24:56.207509: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 14:25:56.346765: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 14:25:56.346933: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7009 - loss: 0.5741 - val_accuracy: 0.7507 - val_loss: 0.5247
Epoch 3/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7162 - loss: 0.55692026-04-10 14:30:00.733722: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 14:30:00.733983: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 14:31:00.968410: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 14:31:00.968640: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 305s 150ms/step - accuracy: 0.7229 - loss: 0.5499 - val_accuracy: 0.7669 - val_loss: 0.5076
Epoch 4/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7303 - loss: 0.54052026-04-10 14:35:05.030201: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 14:35:05.030456: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 14:36:05.189908: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 14:36:05.190014: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7308 - loss: 0.5371 - val_accuracy: 0.7701 - val_loss: 0.4980
Epoch 5/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7337 - loss: 0.53552026-04-10 14:40:08.572124: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 14:40:08.572321: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 14:41:08.731554: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 14:41:08.731715: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7364 - loss: 0.5321 - val_accuracy: 0.7749 - val_loss: 0.4926
Epoch 6/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7342 - loss: 0.53222026-04-10 14:45:12.915923: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 14:45:12.916060: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 14:46:13.075303: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 14:46:13.075555: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7385 - loss: 0.5297 - val_accuracy: 0.7784 - val_loss: 0.4886
Epoch 7/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7356 - loss: 0.53202026-04-10 14:50:17.356886: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 14:50:17.357053: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 14:51:17.601801: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 14:51:17.601916: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 305s 150ms/step - accuracy: 0.7398 - loss: 0.5272 - val_accuracy: 0.7774 - val_loss: 0.4869
Epoch 8/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7384 - loss: 0.52932026-04-10 14:55:21.979663: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 14:55:21.979900: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 14:56:22.120854: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 14:56:22.120995: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 305s 150ms/step - accuracy: 0.7425 - loss: 0.5237 - val_accuracy: 0.7789 - val_loss: 0.4841
Epoch 9/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7390 - loss: 0.52682026-04-10 15:00:26.381510: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:00:26.381700: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 15:01:26.523319: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:01:26.523540: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7414 - loss: 0.5237 - val_accuracy: 0.7809 - val_loss: 0.4830
Epoch 10/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7415 - loss: 0.52622026-04-10 15:05:30.843292: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:05:30.843480: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 15:06:30.984311: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:06:30.984577: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7449 - loss: 0.5218 - val_accuracy: 0.7809 - val_loss: 0.4812
Epoch 11/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7440 - loss: 0.52222026-04-10 15:10:35.404749: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:10:35.404952: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 15:11:35.625441: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:11:35.625616: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 305s 150ms/step - accuracy: 0.7433 - loss: 0.5226 - val_accuracy: 0.7812 - val_loss: 0.4802
Epoch 12/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7441 - loss: 0.52302026-04-10 15:15:40.008588: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:15:40.008838: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 15:16:40.127764: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:16:40.127905: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 305s 150ms/step - accuracy: 0.7457 - loss: 0.5212 - val_accuracy: 0.7824 - val_loss: 0.4790
Epoch 13/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7441 - loss: 0.52262026-04-10 15:20:44.233354: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:20:44.233537: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 15:21:44.391301: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:21:44.391539: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7460 - loss: 0.5209 - val_accuracy: 0.7828 - val_loss: 0.4791
Epoch 14/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7479 - loss: 0.51712026-04-10 15:25:48.571664: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:25:48.571916: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 15:26:48.815797: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:26:48.815943: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7483 - loss: 0.5159 - val_accuracy: 0.7810 - val_loss: 0.4773
Epoch 15/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7436 - loss: 0.52322026-04-10 15:30:52.590426: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:30:52.590634: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 15:31:52.678119: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:31:52.678272: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7453 - loss: 0.5210 - val_accuracy: 0.7818 - val_loss: 0.4766
Epoch 16/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7434 - loss: 0.52282026-04-10 15:35:56.640315: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:35:56.640529: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 15:36:56.774562: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:36:56.774681: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7474 - loss: 0.5185 - val_accuracy: 0.7821 - val_loss: 0.4764
Epoch 17/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7453 - loss: 0.52042026-04-10 15:41:01.157489: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:41:01.157694: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 15:42:01.297834: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:42:01.297917: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 305s 150ms/step - accuracy: 0.7469 - loss: 0.5182 - val_accuracy: 0.7812 - val_loss: 0.4763
Epoch 18/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7456 - loss: 0.51822026-04-10 15:46:05.554422: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:46:05.554620: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 15:47:05.800363: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:47:05.800513: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 305s 150ms/step - accuracy: 0.7467 - loss: 0.5174 - val_accuracy: 0.7835 - val_loss: 0.4754
Epoch 19/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7471 - loss: 0.52062026-04-10 15:51:09.961369: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:51:09.961528: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 15:52:10.102182: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:52:10.102310: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7468 - loss: 0.5189 - val_accuracy: 0.7839 - val_loss: 0.4747
Epoch 20/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7439 - loss: 0.51862026-04-10 15:56:14.005458: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:56:14.005677: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 15:57:14.163294: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 15:57:14.163453: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7459 - loss: 0.5178 - val_accuracy: 0.7820 - val_loss: 0.4752
Epoch 21/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7456 - loss: 0.52072026-04-10 16:01:18.490293: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:01:18.490506: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 16:02:18.645781: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:02:18.646074: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7461 - loss: 0.5191 - val_accuracy: 0.7841 - val_loss: 0.4749
Epoch 22/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7443 - loss: 0.52102026-04-10 16:06:22.867987: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:06:22.868149: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 16:07:23.127614: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:07:23.127746: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 305s 150ms/step - accuracy: 0.7470 - loss: 0.5198 - val_accuracy: 0.7836 - val_loss: 0.4747
Epoch 23/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7424 - loss: 0.52332026-04-10 16:11:27.453547: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:11:27.453767: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 16:12:27.589588: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:12:27.589691: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7459 - loss: 0.5198 - val_accuracy: 0.7828 - val_loss: 0.4751
Epoch 24/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7459 - loss: 0.51952026-04-10 16:16:31.830870: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:16:31.831079: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 16:17:31.973609: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:17:31.973733: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7472 - loss: 0.5178 - val_accuracy: 0.7846 - val_loss: 0.4747
Epoch 25/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7437 - loss: 0.52092026-04-10 16:21:35.994253: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:21:35.994454: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 16:22:36.234644: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:22:36.234725: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7449 - loss: 0.5187 - val_accuracy: 0.7845 - val_loss: 0.4746
Epoch 26/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7443 - loss: 0.52012026-04-10 16:26:40.234052: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:26:40.234238: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 16:27:40.236035: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:27:40.236173: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7464 - loss: 0.5175 - val_accuracy: 0.7848 - val_loss: 0.4737
Epoch 27/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7462 - loss: 0.51902026-04-10 16:31:44.337373: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:31:44.337623: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 16:32:44.478091: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:32:44.478230: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7476 - loss: 0.5177 - val_accuracy: 0.7856 - val_loss: 0.4740
Epoch 28/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7443 - loss: 0.51962026-04-10 16:36:48.660236: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:36:48.660448: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 16:37:48.845530: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:37:48.845648: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7460 - loss: 0.5174 - val_accuracy: 0.7845 - val_loss: 0.4739
Epoch 29/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7443 - loss: 0.52222026-04-10 16:41:52.247865: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:41:52.248056: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 16:42:52.483060: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:42:52.483226: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7451 - loss: 0.5201 - val_accuracy: 0.7868 - val_loss: 0.4734
Epoch 30/30
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 0s 120ms/step - accuracy: 0.7449 - loss: 0.52032026-04-10 16:46:56.568856: W tensorflow/core/framework/local_rendezvous.cc:404] Local
 rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:46:56.569036: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2026-04-10 16:47:56.709160: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
2026-04-10 16:47:56.709298: W tensorflow/core/framework/local_rendezvous.cc:404] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
         [[{{node IteratorGetNext}}]]
         [[IteratorGetNext/_2]]
2000/2000 ━━━━━━━━━━━━━━━━━━━━ 304s 150ms/step - accuracy: 0.7467 - loss: 0.5176 - val_accuracy: 0.7843 - val_loss: 0.4735
Modell sparad
(tf_gpu_env) ubuntu@ip-172-31-40-51:~$ tmux capture-pane -pS - > hela_loggen.txt

