# coding: utf-8
from django.shortcuts import render, render_to_response
from django.http import HttpResponse

import sys
PYTHON_VER = sys.version_info[0]
if PYTHON_VER == '2':
    reload(sys)
    sys.setdefaultencoding('utf-8')

from .info_handler import InfoHandler
from .model_client import ModelClient

# Create your views here.

def assess(request):
	return render(request, 'abroad_assess/assess.html', {})

def result(request):
	import json
	context = {}
	data_legal = False
	def error():
		context['error'] = '数据异常，请重试'
	if request.is_ajax() and request.method == 'POST':
		# for key in request.POST:
		# 	print(key)
		json_str = request.POST.getlist('data')[0]
		# print(json_str)
		if not json_str:
			error()
		try:
			json_data = json.loads(json_str)
			data_legal = True
		except:
			json_data = json.loads(json_str)
			data_legal = True
			error()
	else:
		error()
	if data_legal:
		info_handler = InfoHandler()
		features = info_handler.get_feature(json_data, source='dict')
		model_client = ModelClient()
		result = model_client.predict(features)
                for prob in result:
                    pass
		context['result'] = result
	return render(request, 'abroad_assess/result.html', context)
	# return render_to_response('abroad_assess/assess.html', context)

def test(request):
	return HttpResponse('TEST')
