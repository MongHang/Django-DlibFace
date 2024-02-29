from django.urls import path, re_path
from . import views

urlpatterns = [
    re_path(r'^register/login/$',views.LoginView.as_view(),name='login'),
    re_path(r'^register/$', views.RegisterView.as_view(), name='register'),
    re_path(r'^usernames/(?P<username>[\u4e00-\u9fa5a-zA-Z0-9_-]{1,20})/count/$', views.UsernameCountView.as_view(), name='count_name'),
    re_path(r'^userphones/(?P<phone>09\d{8})/count/$', views.UserphoneCountView.as_view(), name='count_phone'),
    re_path(r'^login/$',views.LoginView.as_view(),name='login'),

]
