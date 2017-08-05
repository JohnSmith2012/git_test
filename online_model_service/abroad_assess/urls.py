from django.conf.urls import url

from . import views

urlpatterns = [
	url(r'^$', views.assess, name='assess'),
	url(r'^result/$', views.result, name='result'),
	url(r'^test/$', views.test, name='test'),
]