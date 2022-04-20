from ml import snds
modeldict={}
train_data = snds(root="../../shapenetcore_partanno_segmentation_benchmark_v0", split= 'train')
train_data.trainsvm(modeldict)
test_data=snds(root="../../shapenetcore_partanno_segmentation_benchmark_v0", split= 'test')
test_data.testsvm(modeldict)