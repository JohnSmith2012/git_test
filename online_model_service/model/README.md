0. 需要安装的库  
	tensorflow  
	$ sudo pip install tensorflow  
	tensorflow serving  
	详见http://tensorflow.github.io/serving/setup  
  
1. 模型的输入  
	模型输入的feature是经过处理的info_handler.py提取的特征（下简称raw_features)。具体对应关系如下  
  
	输入：  
	共30个维度  
	0~11	raw_features的0~11维  
	12～19	学校类型，根据raw_features的第12维one-hot编码得到，共8维  
	20~23	就读阶段，根据raw_features的第13维one-hot编码得到，共4维  
	24	raw_features的14维  
	25	根据raw_features的15维计算得到scaled_gpa  
	26	根据raw_features的16维计算得到scaled_tofel  
	27	根据raw_features的17维计算得到scaled_gre  
	28	根据raw_features的18维计算得到scaled_gmat  
	29	raw_features的19维  
  
	使用原始的gpa, gre, gmat, tofel成绩计算scaled_gpa, scaled_gre, scaled_gmat, scaled_tofel的公式如下：  
	scaled_x = (x - mean)/std  
	if(scaled_x<-3):  
		scaled_x = -1  
	公式中每种成绩对应的参数保存于文件scores_param.txt  
  
2. 样本的标签  
	样本标签使用累积正态分布函数（记作normcdf()）编码，首先计算原始标签录取学校排名的均值mu和方差sigma，然后将排名第i的学校的标签编码为normcdf((j-mu)/sigma)。在只有一个录取学校时不能计算合理的sigma，此时将sigma默认置为30.  
	该标签可认为是申请者申请对应学校成功的概率。  
  
3. 训练和导出模型  
	执行predict_model0730.py，其中编码labels的时间会有点长，大约3~5分钟，训练时间大约3分钟。  
	导出的模型保存在 ./model/00000001/ 中  
  
4. 模型部署  
	部署方法：  
	首先安装tensorflow serving，详见http://tensorflow.github.io/serving/setup。  
  
	安装完成后进入安装目录，执行：  
	$ bazel build //tensorflow_serving/model_servers:tensorflow_model_server  
	$ bazel-bin/tensorflow_serving/model_servers/tensorflow_model_server --port=9000 --model_name=test --model_base_path=$PATH/model0730/model  
	其中$PATH为放置model0730的目录。  
	如无异常输出，表示TensorFlow Serving成功加载了model，可以响应请求。  
  
5. 客户端  
	样例客户端代码和bazel的BUILD文件保存在js_client目录下，使用时将该文件夹复制到安装tensorflow serving的根目录下。  
	在tensorflow serving的根目录下，编译：  
	bazel build //js_client:js_client   
	执行  
	./bazel-bin/js_client/js_client  
	返回结果：  
	outputs {  
	  key: "score"  
	  value {  
	    dtype: DT_FLOAT  
	    tensor_shape {  
	      dim {  
		size: 1  
	      }  
	      dim {  
		size: 394  
	      }  
	    }  
	    float_val: 0.331739336252  
	    float_val: 0.345109343529  
	    float_val: 0.359682679176  
	    float_val: 0.378655254841  
	    float_val: 0.394308239222  
	    float_val: 0.409182935953  
	    float_val: 0.42437556386  
	    float_val: 0.438911885023  
	    float_val: 0.450246334076  
	    float_val: 0.461717039347  
	    float_val: 0.473243057728  
	    float_val: 0.48582804203  
	    float_val: 0.497770398855  
	    float_val: 0.506593167782  
	    float_val: 0.517886698246  
	    float_val: 0.530469298363  
	    float_val: 0.544184982777  
	    float_val: 0.559845626354  
	    float_val: 0.578034698963  
	    ...  
	    float_val: 1.0  
	    float_val: 1.0  
	    float_val: 1.0  
	    float_val: 1.0  
	    float_val: 1.0  
	    float_val: 1.0  
	  }  
	}  
  
