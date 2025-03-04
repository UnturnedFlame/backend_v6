"""
URL configuration for Demo1 project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.views.generic import TemplateView

from app1 import views
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
    TokenVerifyView
)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("user/run_with_datafile_on_cloud/", views.run_model_with_datafile_on_cloud),
    path("shut/", views.shut),
    path("user/save_model/", views.user_save_model),
    path("user/upload_datafile/", views.upload_datafile),
    path("user/delete_datafile/", views.user_delete_datafile),
    path("user/fetch_models/", views.superuser_fetch_models),
    path("user/delete_model/", views.user_delete_model),
    path("user/fetch_datafiles/", views.user_fetch_datafiles),
    path("user/browse_datafile/", views.browse_file_content),
    path("user/user_fetch_private_algorithm", views.user_fetch_private_algorithm),
    path("user/upload_user_private_algorithm/", views.upload_extra_algorithm),
    path("user/user_fetch_extra_algorithm/", views.user_fetch_extra_algorithm),
    path("user/user_feedback/", views.user_feedback),
    path("user/delete_extra_algorithm/", views.delete_extra_algorithm),
    path("user/search_dataset/", views.search_dataset),
    path("user/get_dataset_columns/", views.get_dataset_column_names),
    path("user/save_selected_file/", views.generate_dataset_with_selected_columns),
    path("user/save_union_datafile/", views.concat_dataset),
    path("user/fetch_models_published/", views.user_fetch_models_published),
    path("user/publish_model/", views.user_applies_to_publish_model),
    path("user/generate_conclusion/", views.generate_output),
    path("user/create_component_tree", views.create_component_tree),
    path("user/get_component_trees/", views.get_component_trees),
    path("user/delete_node/", views.delete_node),
    path("user/edit_node_name/", views.edit_node_name),
    path("user/add_node_to_tree/", views.add_node_to_tree),

    path("resetPassword/send_captcha/", views.send_email_captcha),
    path("resetPassword/check_captcha/", views.check_captcha),
    path("resetPassword/reset_password/", views.user_reset_password),

    path('api/token/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/token/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/token/verify/', TokenVerifyView.as_view(), name='token_verify'),
    path('login/', views.login),
    path('captcha/', views.send_email_captcha),
    path('register/', views.register),

    path('administration/fetch_users_models/', views.admin_fetch_users_models),
    path('administration/delete_user_model/', views.admin_delete_model),
    path('administration/fetch_users_info/', views.admin_fetch_users_info),
    path('administration/fetch_users_datafiles/', views.admin_fetch_users_datafiles),
    path('administration/delete_user_datafile/', views.admin_delete_users_files),
    path('administration/reset_user_password/', views.admin_reset_user_password),
    path('administration/add_user/', views.add_user),
    path('administration/delete_user/', views.delete_user),
    path('administration/search_user/', views.search_user),
    path('administration/search_model/', views.search_model),
    path('administration/fetch_feedbacks/', views.fetch_feedbacks),
    path('administration/delete_feedbacks/', views.delete_feedbacks),
    path('administration/update_feedback_status/', views.update_feedback_status),
    path('administration/fetch_publish_model_applications/', views.admin_fetch_publish_model_applications),
    path('administration/handle_publish_model_request/', views.admin_handle_publish_model_application),
    path('administration/delete_publish_model_applications/', views.admin_delete_publish_model_application)

    # path('', TemplateView.as_view(template_name='index.html'), name='index')
]
