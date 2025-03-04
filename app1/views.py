import base64
import json
import multiprocessing
from io import BytesIO
from multiprocessing import freeze_support
import os
import random
import string
import time
from datetime import datetime, timedelta

import jwt
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image as PILImage

from django.contrib.auth import authenticate, get_user_model
from django.contrib.auth.models import Group
from django.core.mail import send_mail
from django.db.models import Q, Case, When, Value, IntegerField
from django.http import JsonResponse, HttpResponse
from matplotlib import pyplot as plt
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Image, SimpleDocTemplate
from reportlab.platypus.flowables import PageBreak

from Backend_v2 import settings
# from app1.module_management.algorithms.functions.speech_processing import extract_speaker_feature
# from app1.module_management.algorithms.functions.load_model import load_model_with_pytorch_lightning
from app1 import models
from app1.models import CaptchaModel
from app1.module_management.calculate_engine import Reactor, encode_image_to_base64
from app1.module_management.algorithms.functions.load_data import load_data

CORS_ALLOW_METHODS = (
    "DELETE",
    "GET",
    "OPTIONS",
    "PATCH",
    "POST",
    "PUT",
)

data_file_root_dir = './app1/recv_file/examples'

# 实例化系统用户对象，使用该对象对数据库中的用户进行相关操作
User = get_user_model()


# 终止运行的操作
def shut(request):
    if request.method == 'GET':
        print('请求shutdown')
        return JsonResponse({'status': 'shutdown'})


# 用户保存模型
def user_save_model(request):
    if request.method == 'POST':
        token = extract_jwt_from_request(request)

        # 所保存模型的基本信息
        model_name = request.POST.get('model_name')
        params_json = request.POST.get('model_info')
        model_description = request.POST.get('description')

        # 保存的模型应该添加到的类型中
        parent_node_value = request.POST.get('parentNode')  # 节点值 value
        treeName = request.POST.get('treeName')  # 树名

        # print('params_json: ', params_json)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')
            user = User.objects.get(username=username)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})
        try:
            component_tree = models.ComponentTree.objects.get(tree_name=treeName)
            # 同一作者的模型不允许重复，同时挂载在同一模型树结构同一节点下的模型名称不允许重复
            parent_node = models.ComponentNode.objects.get(component_tree=component_tree, value=parent_node_value)
            # print('parent_node: ', parent_node.id)

            if models.UserModel.objects.filter(model_name=model_name, author=user).exists():
                return JsonResponse({'message': '保存失败，同名模型已经存在', 'code': 400})
            elif models.ComponentNode.objects.filter(component_tree=component_tree, is_model=True,
                                                     value=model_name,
                                                     parent=parent_node).exists():
                return JsonResponse({'message': '保存失败，该类型下同名模型已经存在', 'code': 400})
            else:

                # 保存的模型默认是未发布的
                saved_model = models.UserModel.objects.create(author=user, model_name=model_name,
                                                              model_info=params_json,
                                                              model_description=model_description)
                # 模型上树
                # 获取父节点
                parent_node = models.ComponentNode.objects.get(component_tree=component_tree, value=parent_node_value)
                parent_node.add_child(label=model_name, is_model=True, user_model=saved_model, is_published='未发布')

                return JsonResponse({'message': '保存模型成功', 'code': 200})
        except Exception as e:
            print("保存模型出错： ", str(e))
            return JsonResponse({'message': str(e), 'code': 400})


# 用户申请发布模型
def user_applies_to_publish_model(request):
    if request.method == 'POST':
        token = extract_jwt_from_request(request)
        model_id = request.POST.get('modelId')

        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})
        try:
            user = User.objects.get(username=username)
            model = models.UserModel.objects.get(id=model_id)
            if model.author == user:
                # 统计该用户申请的发布模型数量，如果超过1000条status为申请中，则不予申请发布
                if models.UserModel.objects.filter(author=user, is_published='申请中').count() > 1000:
                    return JsonResponse({'message': '您的申请已超过1000条，请等待管理员审核', 'code': 400})
                # 向管理员提交发布申请
                models.PublishModelsApplication.objects.create(model=model, applicant=user, create_time=datetime.now())
                # 更新用户已保存模型的发布状态
                model.is_published = '申请中'
                model.save()
                # model.is_publish = True
                # model.save()
                return JsonResponse({'message': '已向管理员提出发布申请，请等待管理员审核', 'code': 200})
            return JsonResponse({'message': '发布失败，您没有发布该模型的权限', 'code': 400})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': '发布失败，请检查模型id是否正确', 'code': 400})


# 管理员获取发布模型的申请
def admin_fetch_publish_model_applications(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})
        try:
            if User.objects.filter(username=username).exists():
                user = User.objects.get(username=username)
                if user.groups.filter(name='admin').exists():
                    # applications = models.PublishModelsApplication.objects.all()
                    applications = models.PublishModelsApplication.objects.annotate(
                        sort_order=Case(
                            When(status='未处理', then=Value(0)),
                            default=Value(1),
                            output_field=IntegerField(),
                        )
                    ).order_by('sort_order', 'create_time')
                    applications_list = []
                    for application in applications:
                        applications_list.append({'id': application.id, 'modelName': application.model.model_name,
                                                  'applicant': application.applicant.username,
                                                  'status': application.status,
                                                  'create_time': application.create_time.strftime('%Y-%m-%d %H:%M:%S'),
                                                  'auditor': application.auditor.username if application.auditor else '未审批',
                                                  'audition_time': application.audition_time.strftime(
                                                      '%Y-%m-%d %H:%M:%S') if application.audition_time else '未审批',
                                                  })
                    return JsonResponse({'message': 'success', 'code': 200, 'data': applications_list})
                return JsonResponse({'message': '您没有权限', 'code': 400})
            return JsonResponse({'message': '用户不存在', 'code': 400})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'unknown error', 'code': 405})


# 管理员删除发布模型申请
def admin_delete_publish_model_application(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)
        application_id = request.GET.get('applicationId')
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            # username = payload.get('username')
            username = payload.get('username')
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})
        try:
            user = User.objects.get(username=username)
            if user.groups.filter(name='admin').exists():
                application = models.PublishModelsApplication.objects.get(id=application_id)
                if application.status == '未处理':
                    if application.model:
                        application.model.is_published = '未发布'
                        application.model.save()
                application.delete()
                return JsonResponse({'message': 'success', 'code': 200})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'unknown error', 'code': 405})


# 管理员处理用户的发布模型申请
def admin_handle_publish_model_application(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)
        application_id = request.GET.get('applicationId')
        status_update = request.GET.get('status')
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            # username = payload.get('username')
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})
        try:
            application = models.PublishModelsApplication.objects.get(id=application_id)
            application.status = status_update
            application.auditor = User.objects.get(username=payload.get('username'))
            application.audition_time = datetime.now()
            application.save()
            print(f'status: {status_update}')

            # 根据状态更新，更新模型的发布状态，模型的发布状态有3种状态，已发布，未发布，申请中
            if status_update == '审核通过':
                application.model.is_published = '已发布'
                application.model.save()
            else:
                application.model.is_published = '未发布'
                application.model.save()
            # 同时更新模型在结构树中的发布状态
            models_on_trees = models.ComponentNode.objects.filter(model=application.model).all()
            for model_on_tree in models_on_trees:
                model_on_tree.is_published = application.model.is_published
                model_on_tree.save()

            return JsonResponse({'message': 'success', 'code': 200})

        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'unknown error', 'code': 405})


# 用户上传新增组件
def upload_extra_algorithm(request):
    if request.method == 'POST':
        token = extract_jwt_from_request(request)
        algorithm_file = request.FILES.get('algorithmFile', None)  # 算法源文件
        model_params_file = request.FILES.get('modelParamsFile', None)  # 算法模型参数文件
        algorithmAlias = request.POST.get('algorithmName', None)  # 上传算法的别名
        statement = request.POST.get('statement', None)  # 上传算法的描述
        algorithm_name = algorithm_file.name  # 所上传算法的文件名

        # algorithm_name = request.POST.get('algorithm_name')  # 算法名
        algorithm_type = request.POST.get('algorithm_type')  # 获取增值服务组件的类型

        # 验证token
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')

            algorithm_type_mapping = {'插值处理': "private_interpolation", '特征选择': 'extra_feature_selection',
                                      '特征提取': 'private_feature_extraction', '无量纲化': 'private_scale',
                                      '小波变换': 'extra_wavelet_transform', '故障诊断': 'private_fault_diagnosis',
                                      '故障预测': 'private_fault_prediction', '健康评估': 'extra_health_evaluation'}

            algorithm_type = algorithm_type_mapping.get(algorithm_type)
            # 根据使用的是机器学习还是深度学习的故障诊断进行分类
            if algorithm_type == 'private_fault_diagnosis':
                faultDiagnosisType = request.POST.get('faultDiagnosisType')
                if faultDiagnosisType == 'machineLearning':
                    algorithm_type = 'private_fault_diagnosis_ml'
                else:
                    algorithm_type = 'private_fault_diagnosis_dl'
            if algorithm_file is not None:
                # 创建用户私有算法的目录
                user_save_dir = f"app1/module_management/algorithms/models/{algorithm_type}/{username}"
                if not os.path.exists(user_save_dir):
                    os.makedirs(user_save_dir)

                # 保存用户私有算法
                algorithm_save_path = user_save_dir + "/" + algorithm_name
                # 如果存在重名文件，在新的文件名中添加时间戳
                while os.path.exists(algorithm_save_path):
                    algorithm_name = algorithm_name.split('.')[0] + '_' + str(
                        datetime.now().strftime("%Y%m%d%H%M%S")) + '.' + algorithm_name.split('.')[1]
                    algorithm_save_path = user_save_dir + "/" + algorithm_name

                # 保存用户专有算法的模型参数
                isFaultDiagnosis = 'private_fault_diagnosis' in algorithm_type
                isFaultPrediction = 'private_fault_prediction' in algorithm_type
                isHealthEvaluation = 'extra_health_evaluation' in algorithm_type
                # 如果是故障诊断、故障预测或是健康评估，则需要保存模型参数
                if isFaultDiagnosis or isFaultPrediction or isHealthEvaluation:
                    model_file_name = algorithm_name.split('.')[0] + '.' + model_params_file.name.split('.')[-1]
                    model_params_save_path = user_save_dir + "/" + model_file_name  # 私有故障诊断算法的模型参数保存路径
                    with open(model_params_save_path, 'wb+') as f:
                        # 分块写入文件
                        for chunk in model_params_file.chunks():
                            f.write(chunk)
                else:
                    model_params_save_path = ''

                # 写入数据库
                user = User.objects.get(username=username)
                # if not models.PrivateAlgorithmOfUser.objects.filter(owner=user,
                #                                                     algorithm_name=algorithm_name).exists():
                if not models.PrivateAlgorithmOfUser.objects.filter(alias=algorithmAlias).exists():
                    with open(algorithm_save_path, 'wb+') as f:
                        for chunk in algorithm_file.chunks():
                            f.write(chunk)
                    saved_algorithm = models.PrivateAlgorithmOfUser.objects.create(owner=user,
                                                                                   algorithm_name=algorithm_name,
                                                                                   alias=algorithmAlias,
                                                                                   statement=statement,
                                                                                   algorithm_type=algorithm_type,
                                                                                   file_path=algorithm_save_path,
                                                                                   model_filepath=model_params_save_path)
                    saved_algorithm.save()
                    # print('算法上传成功')
                    return JsonResponse({'message': '用户私有算法上传成功', 'code': 200})
                else:
                    return JsonResponse({'message': '同名增值服务组件已存在', 'code': 400})
            return JsonResponse({'message': '无效的文件路径', 'code': 404})

        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})


# 开发者用户获取已上传的增值服务组件
def user_fetch_extra_algorithm(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)
        keyword = request.GET.get('keyword', None)
        try:
            # 验证token
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')
            print(f'username: {username}')
            algorithmTypeMapping = {'private_interpolation': '插值处理', 'extra_feature_selection': '特征选择',
                                    'private_feature_extraction': '特征提取', 'private_scale': '无量纲化',
                                    'extra_wavelet_transform': '小波变换', 'private_fault_diagnosis_ml': '故障诊断',
                                    'private_fault_diagnosis_dl': '故障诊断',
                                    'private_fault_prediction': '故障预测', 'extra_health_evaluation': '健康评估'}
            user = User.objects.get(username=username)
            if not keyword:
                objects = models.PrivateAlgorithmOfUser.objects.filter(owner=user)
            else:
                objects = models.PrivateAlgorithmOfUser.objects.filter(owner=user, alias__icontains=keyword)
            posts = [
                {'algorithmType': algorithmTypeMapping[post.algorithm_type],
                 'algorithmName': post.algorithm_name,
                 'alias': post.alias,
                 'statement': post.statement,
                 'id': post.id,
                 'machineLearning': 'ml' if 'private_fault_diagnosis_ml' in post.algorithm_type else 'dl'}
                for post in objects]
            # print('posts:', posts)
            return JsonResponse({'message': 'user fetch extra algorithm success', 'code': 200, 'data': posts})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})


# 用户反馈
def user_feedback(request):
    if request.method == 'POST':
        token = extract_jwt_from_request(request)
        model_name = request.POST.get('modelName')
        module = request.POST.get('module')
        question = request.POST.get('question')
        data_filename = request.POST.get('datafile')
        model_id = request.POST.get('modelId')
        print(f'model_id: {model_id}')

        # 打印接收到的表单数据
        print(f'model_name: {model_name}')
        print(f'module: {module}')
        print(f'question: {question}')
        print(f'data_filename: {data_filename}')

        try:
            # 验证token
            try:
                payload = verify_jwt(token, settings.SECRET_KEY)
                username = payload.get('username')
                print(f'username: {username}')
            except Exception as e:
                print(str(e))
                return JsonResponse({'message': str(e), 'code': 401})
            user = User.objects.get(username=username)
            # 存储用户的反馈
            try:
                if model_id and models.UserModel.objects.filter(id=model_id).exists():

                    models.Feedback.objects.create(
                        user=user,
                        model=models.UserModel.objects.get(id=model_id),
                        datafile=models.SavedDatasetsFromUser.objects.get(dataset_name=data_filename, owner=user),
                        module=module,
                        question=question
                    )
                    return JsonResponse({'message': '反馈已提交', 'code': 200})

            except Exception as e:
                print(str(e))
                return JsonResponse({'message': '创建用户反馈失败', 'code': 405})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'unknown error', 'code': 405})


# 管理员查看用户反馈
def fetch_feedbacks(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})
        try:
            # feedbacks = models.Feedback.objects.all().order_by(
            #
            # )
            feedbacks = models.Feedback.objects.annotate(
                sort_order=Case(
                    When(status='待处理', then=Value(0)),
                    default=Value(1),
                    output_field=IntegerField(),
                )
            ).order_by('sort_order', 'create_time')
            posts = [{'id': feedback.id, 'question': feedback.question, 'datafile': feedback.datafile.dataset_name,
                      'model': feedback.model.model_name, 'module': feedback.module,
                      'model_author': feedback.model.author.username,
                      'time': feedback.create_time.strftime('%Y-%m-%d %H:%M:%S'),
                      'username': feedback.user.username, 'status': feedback.status} for feedback in feedbacks]
            print('posts:', posts)
            return JsonResponse({'message': '获取反馈成功', 'code': 200, 'data': posts})

            # return render(request, 'manager_feedback.html', {'feedbacks': feedbacks})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'unknown error', 'code': 405})


def delete_feedbacks(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)
        feedback_id = request.GET.get('feedbackId')

        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})

        try:
            models.Feedback.objects.get(id=feedback_id).delete()
            return JsonResponse({'message': '删除反馈成功', 'code': 200})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': '删除反馈失败', 'code': 405})


# 删除增值服务组件
def delete_extra_algorithm(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)
        algorithmAlias = request.GET.get('algorithmAlias')
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})
        user = User.objects.get(username=username)
        if models.PrivateAlgorithmOfUser.objects.filter(alias=algorithmAlias).exists():
            algorithm = models.PrivateAlgorithmOfUser.objects.get(alias=algorithmAlias)
            algorithm_filepath = algorithm.file_path
            model_filepath = algorithm.model_filepath
            if algorithm.owner == user:
                # 如果是用户私有算法，则删除文件
                algorithm.delete()  # 删除算法源文件
                os.remove(algorithm_filepath)
                if os.path.isfile(model_filepath):
                    if os.path.exists(model_filepath):
                        os.remove(model_filepath)  # 删除算法模型文件
                return JsonResponse({'message': 'user delete extra algorithm success', 'code': 200})
            else:
                return JsonResponse({'message': '没有权限删除该组件', 'code': 400})
        else:
            return JsonResponse({'message': '算法不存在', 'code': 404})


def search_user(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)
        keywordType = request.GET.get('keywordType')
        keywords = request.GET.get('keywords')

        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 401})

        try:

            if keywordType == 'username':
                users = User.objects.filter(username__icontains=keywords)
            else:
                users = User.objects.filter(jobNumber__icontains=keywords)
            result = [
                {
                    'id': user.id,
                    'username': user.username,
                    'jobNumber': user.jobNumber,
                    'email': user.email,

                } for user in users
            ]
            return JsonResponse({'message': '搜索用户成功', 'code': 200, 'result': result})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'unknown error', 'code': 405})


def search_model(request):
    if request.method == 'GET':
        keywordType = request.GET.get('keywordType')
        keywords = request.GET.get('keywords')

        token = extract_jwt_from_request(request)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 401})
        try:
            if keywordType == 'username':
                objects = models.UserModel.objects.filter(author__username__icontains=keywords)
            else:
                objects = models.UserModel.objects.filter(model_name__icontains=keywords)
            result = [
                {
                    'id': obj.id,
                    'author': obj.author.username,
                    'model_name': obj.model_name,
                    'model_info': obj.model_info,
                    'jobNumber': obj.author.jobNumber,
                } for obj in objects
            ]
            return JsonResponse({'message': '搜索模型成功', 'code': 200, 'result': result})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'unknown error', 'code': 405})


# 用户配置私有算法时，访问该接口，接口返回该用户所有私有算法
def user_fetch_private_algorithm(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)
        algorithm_type = request.GET.get('algorithm_type')  # 私有算法的类型
        # algorithm_name = request.GET.get('algorithm_name')  # 私有算法的算法名
        # 确定用户所配置的私有算法的算法类型
        try:
            # 验证token
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')

            # 返回给该用户其所有的某个特定类型的全部私有算法的算法名
            # user = User.objects.get(username=username)
            # objects = models.PrivateAlgorithmOfUser.objects.filter(owner=user, algorithm_type=algorithm_type)
            objects = models.PrivateAlgorithmOfUser.objects.filter(algorithm_type=algorithm_type)
            if objects:
                private_algorithms = [{'algorithmAlias': obj.alias, 'algorithmStatement': obj.statement}
                                      for obj in objects]
            else:
                private_algorithms = []
            # print('private_algorithms:', private_algorithms)
            return JsonResponse({'algorithmList': private_algorithms, 'code': 200})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 400})


# 开发者用户通过用户名获取已保存的模型，包括已发布的和未发布的模型
def superuser_fetch_models(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)

        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 401})

        try:
            username = payload.get('username')
            user = User.objects.get(username=username)
            # user = User.objects.get(username=username)
            # objects = models.SavedModelFromUser.objects.filter(author=user)
            objects = models.UserModel.objects.filter(author=user)
            # posts = objects.values()
            posts = [{
                'id': obj.id,
                'author': obj.author.username,
                'model_name': obj.model_name,
                'model_info': obj.model_info,
                'description': obj.model_description,
                'isPublish': obj.is_published
            } for obj in objects]

            return JsonResponse({'models': posts, 'code': 200}, safe=False)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})


# 用户访问开发者用户发布的模型
def user_fetch_models_published(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)
        try:
            _ = verify_jwt(token, settings.SECRET_KEY)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 401})
        try:
            # objects = models.SavedModelFromUser.objects.filter(is_published='已发布')
            objects = models.UserModel.objects.filter(is_published='已发布')
            posts = [{
                'id': obj.id,
                'author': obj.author.username,
                'model_name': obj.model_name,
                'model_info': obj.model_info,
                'description': obj.model_description,
                'isPublish': obj.is_published
            } for obj in objects]

            return JsonResponse({'models': posts, 'code': 200}, safe=False)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})


# 管理员获取所有用户的模型
def admin_fetch_users_models(request):
    if request.method == 'GET':
        all_user_models = models.UserModel.objects.all()
        token = extract_jwt_from_request(request)

        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')

            if not is_administrator(username):
                return JsonResponse({'message': '没有权限查看用户模型信息', 'code': 400})

            models_list = [
                {
                    'id': model.id,
                    'model_name': model.model_name,
                    'model_info': model.model_info,
                    'description': model.model_description,
                    'author': model.author.username,
                    'jobNumber': model.author.jobNumber,
                    'isPublished': model.is_published
                } for model in all_user_models
            ]

            return JsonResponse(models_list, safe=False)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})


# 管理员获取所有用户信息
def admin_fetch_users_info(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')
            user = User.objects.get(username=username)

            admin_group = Group.objects.filter(name='admin').first()
            if admin_group in user.groups.all():
                all_users = User.objects.all().values()
                return JsonResponse(list(all_users), safe=False)
            else:
                print('没有访问权限')
                return JsonResponse({'message': '没有访问权限', 'code': 400})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})


# 管理员重置用户密码
def admin_reset_user_password(request):
    if request.method == 'GET':
        jobNumber = request.GET.get('jobNumber')
        token = extract_jwt_from_request(request)

        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')

            if not is_administrator(username):
                print('没有权限重置用户密码')
                return JsonResponse({'message': '没有权限重置用户密码', 'code': 400})
            user = User.objects.filter(jobNumber=jobNumber).first()

            if user:
                # 生成8位随机数作为新密码
                length = 8
                characters = string.ascii_letters + string.digits  # 大小写字母和数字
                new_password = ''.join(random.choice(characters) for _ in range(length))
                # 重置用户密码
                user.set_password(new_password)
                user.save()

                return JsonResponse({'message': f'密码重置成功，新密码为{new_password}', 'code': 200})
            else:
                return JsonResponse({'message': '重置密码失败, 未找到该用户', 'code': 404})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})


# 管理员删除用户模型
def admin_delete_model(request):
    if request.method == 'GET':
        row_id = request.GET.get('row_id')
        token = extract_jwt_from_request(request)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload['username']
            # 管理员才能删除用户模型
            if not is_administrator(username):
                print('无权删除用户模型')
                return JsonResponse({'message': '没有权限删除用户模型', 'code': 400})

            models.UserModel.objects.filter(id=row_id).delete()
            return JsonResponse({'message': 'delete user model success', 'code': 200})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})


# 管理员获取用户数据文件
def admin_fetch_users_datafiles(request):
    if request.method == 'GET':
        all_data_files = models.SavedDatasetsFromUser.objects.all()
        token = extract_jwt_from_request(request)

        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload['username']
            user = User.objects.get(username=username)
            # 只有管理员可以访问用户列表
            admin_group = Group.objects.filter(name='admin').first()
            if admin_group not in user.groups.all():
                print('没有权限删除用户数据')
                return JsonResponse({'message': '没有权限删除用户数据', 'code': 400})

            posts = [
                {
                    'id': file_info.id,
                    'owner': file_info.owner.username,
                    'dataset_name': file_info.dataset_name,
                    'description': file_info.description
                } for file_info in all_data_files
            ]

            return JsonResponse(posts, safe=False)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})


# 管理员删除用户数据文件
def admin_delete_users_files(request):
    if request.method == 'GET':
        file_id = request.GET.get('datafile_id')
        token = extract_jwt_from_request(request)
        # print('token: ', token)
        # print('datafile_id: ', file_id)

        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            user = User.objects.get(username=payload.get('username'))
            admin_group = Group.objects.filter(name='admin').first()

            if admin_group not in user.groups.all():
                print('没有权限删除用户数据')
                return JsonResponse({'message': '没有权限删除用户数据', 'code': 400})

            datafile = models.SavedDatasetsFromUser.objects.filter(id=file_id).first()

            if datafile:
                try:
                    file_path = datafile.file_path
                    # print('file_path: ', file_path)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                    datafile.delete()
                except Exception as e:
                    print(str(e))
                    return JsonResponse({'message': str(e), 'code': 400})
                return JsonResponse({'message': '删除用户数据成功', 'code': 200})
            else:
                return JsonResponse({'message': '找不到对应文件', 'code': 404})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})


# 用户删除模型
def user_delete_model(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)
        row_id = request.GET.get('row_id')
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            try:
                user = User.objects.get(username=payload.get('username'))
                count, _ = models.UserModel.objects.filter(author=user, id=row_id).delete()
                if count == 0:
                    return JsonResponse({'message': '没有权限删除模型', 'code': 404})
                return JsonResponse({'message': ''
                                                '', 'code': 200})
            except Exception as e:
                return JsonResponse({'message': str(e), 'code': 400})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})


# 用户上传文件到服务器，上传数据文件存储到服务器
def upload_datafile(request):
    if request.method == 'POST':
        datafile = request.FILES.get('file', None)
        filename = request.POST.get('filename')
        description = request.POST.get('description')
        multiple_sensors = request.POST.get('multipleSensors')

        # is_public = request.POST.get('isPublic')
        # print(f'is_public: {is_public}')
        # public = True if is_public == 'public' else False

        # if multiple_sensors == 'multiple':
        #     multiple_sensors = True
        # else:
        #     multiple_sensors = False
        multiple_sensor = True if multiple_sensors == 'multiple' else False
        # print("datafile: ", datafile)
        # print("filename: ", filename)
        # print("description: ", description)
        token = extract_jwt_from_request(request)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')

            if datafile is not None:
                user_save_dir = f"./app1/recv_file/examples/{username}"
                if not os.path.exists(user_save_dir):
                    os.makedirs(user_save_dir)
                # 获取文件类型
                filetype = datafile.name.split('.')[-1]
                save_path = user_save_dir + "/" + filename + '.' + filetype
                # print(f'save path: {save_path}')
                user = User.objects.get(username=username)
                if not models.SavedDatasetsFromUser.objects.filter(owner=user,
                                                                   dataset_name=filename).exists():
                    with open(save_path, 'wb+') as f:
                        # 分块写入文件
                        for chunk in datafile.chunks():
                            f.write(chunk)
                    saved_data = models.SavedDatasetsFromUser.objects.create(owner=user,
                                                                             dataset_name=filename,
                                                                             description=description,
                                                                             file_path=save_path,
                                                                             multiple_sensors=multiple_sensor,
                                                                             file_type=filetype,
                                                                             )
                    saved_data.save()
                    return JsonResponse({'message': 'save data success', 'code': 200})
                else:
                    return JsonResponse({'message': '同名文件已存在', 'code': 400})
            return JsonResponse({'message': '无效的文件路径', 'code': 400})

        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})


# 用户获取上传的文件数据
def user_fetch_datafiles(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)
        public_only = request.GET.get('publicOnly', None)
        public_only = True if public_only == 'Y' else False
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})
        username = payload.get('username')
        try:
            # print(f'接收到来自用户{username}的获取数据请求')
            user = User.objects.get(username=username)
            if not public_only:
                objects = models.SavedDatasetsFromUser.objects.filter(Q(owner=user) | Q(publicity=True))
            else:
                objects = models.SavedDatasetsFromUser.objects.filter(publicity=True)

            posts = [
                {
                    'id': obj.id,
                    'dataset_name': obj.dataset_name,
                    'description': obj.description,
                    'owner': obj.owner.username,
                    'file_type': obj.file_type
                } for obj in objects
            ]

            return JsonResponse(posts, safe=False)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 405})


# 用户删除数据文件
def user_delete_datafile(request):
    if request.method == 'GET':
        token = extract_jwt_from_request(request)
        filename = request.GET.get('filename')
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')
            user = User.objects.get(username=username)
            objects = models.SavedDatasetsFromUser.objects.filter(owner=user, dataset_name=filename).first()
            if objects:
                file_path = objects.file_path
                os.remove(file_path) if os.path.exists(file_path) else None
                objects.delete()

                return JsonResponse({'message': 'deleted successfully', 'code': 200})
            else:
                return JsonResponse({'message': '没有权限删除该数据文件', 'code': 400})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})


example_for_validation = {
    'single_sensor_example': 'app1/recv_file/examples/validation_examples/single_example.npy',
    'multiple_sensor_example': 'app1/recv_file/examples/validation_examples/multiple_example.npy',
    # 'example_for_interpolation_validation': 'app1/recv_file/examples/validation_examples/data_for_interpolation_validation.npy',
    'example_for_interpolation_validation': 'app1/recv_file/examples/validation_examples/sampled_example.npy',
    'example_for_fault_diagnosis_validation': 'app1/recv_file/examples/validation_examples/exampleForFaultDiagnosisValidation.npy',
    'example_for_fault_diagnosis_validation_multiple': 'app1/recv_file/examples/validation_examples/multiple_sensors_7_dataset_10.npy',
    'labels': 'app1/recv_file/examples/validation_examples/exampleForFaultDiagnosisValidation_label.npy',
    'multiple_sensor_labels': 'app1/recv_file/examples/validation_examples/multiple_sensors_7_labels_10.npy',
}

user_recv_data_dir = 'app1/recv_file/examples'


# 通过服务器中保存的用户数据文件运行
def run_model_with_datafile_on_cloud(request):
    if request.method == 'POST':
        # print("收到运行模型的请求")
        token = extract_jwt_from_request(request)
        file_name = request.POST.get('file_name', None)  # 校验之外的模型运行使用数据库中存放的文件
        # public_data = request.POST.get('public_data', None)
        validation_filename = request.POST.get('validationExample', None)
        validation_file = request.FILES.get('validationExampleFile', None)

        # print('token: ', token)
        # print('file_name: ', file_name)
        try:
            try:
                payload = verify_jwt(token, settings.SECRET_KEY)
                username = payload.get('username')
            except Exception as e:
                # 根据不同的异常类型执行不同的逻辑
                if str(e) == 'JWT已过期':
                    print("错误: JWT已过期，请重新登录。")
                elif str(e) == '无效的JWT':
                    print("错误: 无效的JWT，请检查您的凭证。")
                else:
                    print(f"未知错误: {e}")
                return JsonResponse({'message': 'token invalid', 'code': 401})

            # 查询savedDatasetFromUserFromUser表中，获取文件路径
            user = User.objects.get(username=username)
            if file_name is not None:
                multiple = models.SavedDatasetsFromUser.objects.filter(owner=user,
                                                                       dataset_name=file_name).first().multiple_sensors
                user = User.objects.get(username=payload.get('username'))
                file = models.SavedDatasetsFromUser.objects.filter(owner=user, dataset_name=file_name).first()

                if file is not None:
                    file_path = file.file_path
                else:
                    # 查询结果为空
                    print('未知的数据文件')
                    return JsonResponse({'message': '找不到指定的数据文件', 'code': 404})
            else:
                # 使用校验样本对上传的增值服务组件进行完整性校验

                if validation_filename == 'single_sensor_example':
                    multiple = False
                    example_filepath = example_for_validation['single_sensor_example']
                elif validation_filename == 'example_for_interpolation_validation':
                    multiple = False
                    example_filepath = example_for_validation['example_for_interpolation_validation']  # 样本文件路径

                elif validation_filename == 'example_for_fault_diagnosis_validation':
                    multiple = False
                    example_filepath = example_for_validation['example_for_fault_diagnosis_validation']
                elif validation_filename == 'example_for_fault_diagnosis_validation_multiple':
                    multiple = True
                    example_filepath = example_for_validation['example_for_fault_diagnosis_validation_multiple']
                else:
                    multiple = True
                    example_filepath = example_for_validation['multiple_sensor_example']

                example_filename = os.path.basename(example_filepath)
                file_path = user_recv_data_dir + '/' + username + '/' + example_filename  # 用户调用目录下的样本保存路径
                print(f'file_path: {file_path}')
                # 分块读取并保存
                buffer_size = 1024 * 1024  # 1MB buffer size
                with open(file_path, 'wb') as destination:
                    with open(example_filepath, 'rb') as source:
                        while True:
                            data = source.read(buffer_size)
                            if not data:
                                break
                            destination.write(data)
            # print(f'multiple_sensors: {multiple}')
            # print('payload: ', payload)
            params = json.loads(request.POST.get('params'))

            algorithm_dict = params['algorithms']
            params_dict = params['parameters']
            schedule = params['schedule'][1:]
            # multiple_sensor = params['multipleSensor']

            print('algorithm_dict: ', algorithm_dict)
            print('params_dict: ', params_dict)
            print('schedule: ', schedule)

            # 当运行的算法为增值服务组价的算法时，需要将算法的参数转变为存放该算法源文件的路径
            for k, v in params_dict.items():
                if 'private_' in k or 'extra_' in k:
                    # 从数据库中获取该算法的源文件路径，并作为参数传入算法引擎
                    if k != 'private_scaler':
                        algorithm_name = v
                    else:
                        algorithm_name = v['algorithm']
                    print(f'k: {k}, v: {v}')
                    obj = models.PrivateAlgorithmOfUser.objects.filter(alias=algorithm_name).first()
                    if obj is None:
                        print(f'{algorithm_name}算法不存在')
                        return JsonResponse({'message': '算法不存在', 'code': 404})
                    algorithm_path = obj.file_path  # 算法源文件路径
                    print('algorithm_path: ', algorithm_path)
                    if k == 'private_scaler':
                        params_dict[k]['algorithm'] = algorithm_path
                    else:
                        params_dict[k] = algorithm_path

            try:
                # 初始化算法引擎
                demo_app = Reactor(schedule, algorithm_dict, params_dict, multiple)
                print('算法引擎初始化成功......')
                # demo_app.init(schedule, algorithm_dict, params_dict)
            except Exception as e:
                print(str(e))
                print('算法引擎初始化错误......')
                return JsonResponse({'message': '算法引擎初始化错误', 'code': 400})

            try:
                if os.path.exists(file_path):
                    # 存放子线程返回结果的队列
                    # freeze_support()
                    # queue = multiprocessing.Queue()
                    # new_subprocess = multiprocessing.Process(target=demo_app.start, args=(file_path, queue))
                    # new_subprocess.start()
                    # new_subprocess.join()
                    # results: dict = queue.get()
                    demo_app.start(datafile=file_path)
                    error = demo_app.error
                    print(f"error : {error}")
                    if error:
                        return JsonResponse({'message': error, 'code': 400})
                    results = demo_app.results_to_response
                    # print(f"results: {results}")
                    for module in schedule:
                        if not results[module]:
                            return JsonResponse({'message': '数据匹配出错', 'code': 400}, status=400)

                    # 根据返回的图片路径读取结果图片，然后编码为Base64字符串之后返回给前端
                    result_figure_of_health_evaluation = ['二级指标权重柱状图_Base64', '评估结果柱状图_Base64',
                                                          '层级有效指标_Base64']
                    for module, result in results.items():
                        # print(f"module: {module}, result: {result}")
                        if module == '故障诊断' and result and validation_filename is not None:
                            # 绘制故障诊断校验结果
                            num_has_fault_prediction = result['num_has_fault']
                            # num_has_no_fault_prediction = result['num_has_no_fault']
                            if validation_filename == 'example_for_fault_diagnosis_validation':
                                labels = np.load(example_for_validation['labels'])
                            else:
                                labels = np.load(example_for_validation['multiple_sensor_labels'])
                            num_fault_truth = np.sum(labels).item()
                            example_num = len(labels)
                            # 绘制故障诊断结果对比图
                            fd_validation_result = plot_diagnosis_result_comparison(file_path, num_fault_truth,
                                                                                    num_has_fault_prediction,
                                                                                    example_num)
                            # num_no_fault_truth = len(labels) - num_fault_truth
                            # 添加到结果中
                            results[module]['fd_validation_result'] = fd_validation_result
                        if module == '健康评估' and result and validation_filename is not None:
                            if validation_filename == 'example_for_fault_diagnosis_validation':
                                labels = np.load(example_for_validation['labels'])
                            else:
                                labels = np.load(example_for_validation['multiple_sensor_labels'])
                            sample_state_membership = result['各样本状态隶属度']
                            print(f"各样本状态隶属度: {sample_state_membership}")
                            he_validation_result = plot_health_evaluation_comparison(file_path, sample_state_membership,
                                                                                     labels)
                            results[module]['he_validation_result'] = he_validation_result

                        for key, value in result.items():
                            if 'Base64' in key:
                                # 健康评估中的可视化的结果是每个样本都有一组
                                if key in result_figure_of_health_evaluation:
                                    # print(f'value: {value}')
                                    # res_list = value
                                    for i, res in enumerate(value):
                                        result[key][i] = encode_image_to_base64(res)
                                else:
                                    result[key] = encode_image_to_base64(value)
                    # print('results_after: ', results)
                    return JsonResponse({'message': 'success', 'results': results, 'code': 200})
                else:
                    return JsonResponse({'message': 'fail, file not found', 'code': 404})
            except Exception as e:
                print(f"运行算法引擎出错： ", str(e))
                return JsonResponse({'message': '数据匹配出错', 'code': 404})

        except Exception as e:
            print(str(e))
            return JsonResponse({'message': '未知错误', 'code': 402})


# def login_user(request):
#     if request.method == 'POST':
#         username = request.GET.get('username')
#         password = request.GET.get('password')
#
#         user = User.objects.filter(username=username).first()
#         if not user:
#             return JsonResponse({'message': 'user not exists'})
#         # 获取用户角色
#         user_groups = user.groups.all()
#         # 检查用户角色
#         specific_group_name = 'User'
#         user_is_in_specific_group = any(group.name == specific_group_name for group in user_groups)
#         if not user_is_in_specific_group:
#             return JsonResponse({'message': 'user not exists'})
#
#         # 检查登录密码
#         if password == user.password:
#             return JsonResponse({'message': 'login success'})
#         else:
#             return JsonResponse({'message': 'invalid password'})


# 绘制诊断组件校验结果比对图
def plot_diagnosis_result_comparison(example_filepath, num_fault_truth, num_fault_prediction, example_num):
    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    split_filepath = example_filepath.split('/')
    user_dir = split_filepath[-2] + '/' + split_filepath[-1].split('.')[0]
    # 数据准备
    categories = ['故障', '无故障']
    num_no_fault_truth = example_num - num_fault_truth
    num_no_fault_prediction = example_num - num_fault_prediction
    truth_values = [num_fault_truth, num_no_fault_truth]
    prediction_values = [num_fault_prediction, num_no_fault_prediction]

    # 设置柱状图的位置
    x = np.arange(len(categories))  # the label locations
    width = 0.35  # the width of the bars

    # 创建图形和轴
    fig, ax = plt.subplots()

    # 绘制柱状图
    rects1 = ax.bar(x - width / 2, truth_values, width, label='真实样本数量', color='skyblue')
    rects2 = ax.bar(x + width / 2, prediction_values, width, label='预测样本数量', color='salmon')

    # 添加一些文本标签
    ax.set_xlabel('类别')
    ax.set_ylabel('数量')
    ax.set_title('预测结果对比图')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    save_path_dir = 'app1/module_management/algorithms/functions/validation_results/' + user_dir
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    save_path = save_path_dir + '/' + f"{split_filepath[-1].split('.')[0]}_waveform.png"
    # 保存图像
    plt.savefig(save_path)
    plt.close()

    # 返回对比图的Base64编码
    figure_Base64 = encode_image_to_base64(save_path)

    return figure_Base64


def plot_health_evaluation_comparison(example_filepath, sample_state_membership: dict, labels):
    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    split_filepath = example_filepath.split('/')
    user_dir = split_filepath[-2] + '/' + split_filepath[-1].split('.')[0]

    # # 提取各样本状态隶属度
    # sample_state_membership: dict

    # 提取各标签样本数量
    labels = labels.flatten()
    num_has_fault = np.sum(labels).item()
    num_no_fault = labels.shape[0] - num_has_fault

    label_counts = np.array([num_has_fault, num_no_fault])

    # print(f"label_counts: {label_counts}")

    # 设置中文字体
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    # 绘制饼状图
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

    # 饼状图
    axes[0].pie(sample_state_membership.values(),
                colors=colors,
                labels=sample_state_membership.keys(),
                autopct='%1.1f%%', startangle=90)
    axes[0].set_title('各样本状态隶属度')
    axes[0].legend(title="状态")

    # 柱状图
    axes[1].bar(['无故障', '有故障'], label_counts, color=['gray', 'orange'])
    axes[1].set_title('测试样本的各标签样本数量')
    axes[1].set_ylabel('数量')
    axes[1].legend(['标签'])

    plt.tight_layout()

    save_path_dir = 'app1/module_management/algorithms/functions/validation_results/' + user_dir
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    save_path = save_path_dir + '/' + f"{split_filepath[-1].split('.')[0]}_health_evaluation.png"
    # 保存图像
    plt.savefig(save_path)
    plt.close()

    # 返回对比图的Base64编码
    figure_Base64 = encode_image_to_base64(save_path)

    return figure_Base64


# 绘制信号波形图
def plot_raw_data_waveform(example_filepath):
    """
    绘制信号的波形图
    :param example_filepath: 输入信号
    :return: 绘制的图形的保存路径
    """
    # 获取用户目录
    split_filepath = example_filepath.split('/')
    user_dir = split_filepath[-2] + '/' + split_filepath[-1].split('.')[0]

    example, filename = load_data(example_filepath)
    # example = example.reshape(2048, -1)

    matplotlib.use('Agg')
    # 设置字体以支持中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号 '-' 显示为方块的问题
    plt.rcParams['font.size'] = 15

    if len(example.shape) == 1:
        example = example.reshape(-1, 1)

    # 获取传感器的数量
    shape = example.shape
    # 确保输入信号的形状为(2048, N)，其中N为传感器的数量，默认传感器的数量是要小于采样点的
    if shape[0] < shape[1]:
        example = example.T

    num_sensor = example.shape[1]

    if num_sensor == 1:
        # 单传感器的数据
        plt.figure(figsize=(16, 8))

        plt.plot(example.flatten())
        plt.title('信号波形图')

        plt.xlabel('采样点')
        plt.ylabel('信号值')
    else:
        plt.figure(figsize=(20, 15))

        # 创建图形和子图

        for i in range(num_sensor):

            plt.subplot(num_sensor, 1, i + 1)  # 创建子图，num_sensors 行 1 列，当前是第 i+1 个子图
            plt.plot(example[:, i])  # 绘制第 i 个传感器的信号
            plt.title(f'传感器{i + 1}')  # 设置子图的标题
            plt.xlabel('采样点', )  # 设置 x 轴标签
            plt.ylabel('信号值', )  # 设置 y 轴标签
            if i + 1 >= num_sensor:
                break

        plt.tight_layout()

    # 保存图像
    save_path_dir = 'app1/module_management/algorithms/functions/raw_data/' + user_dir
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)
    save_path = save_path_dir + '/' + f'{filename}_waveform.png'
    plt.savefig(save_path)

    # 返回信号波形图的Base64编码
    figure_Base64 = encode_image_to_base64(save_path)

    return figure_Base64


# 用户查看文件的内容
def browse_file_content(request):
    if request.method == 'GET':
        filename = request.GET.get('filename')
        # print(f'filename: {filename}')
        token = extract_jwt_from_request(request)

        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')

            user = User.objects.get(username=username)

            file_object = models.SavedDatasetsFromUser.objects.filter(owner=user, dataset_name=filename).first()
            if not file_object:
                print('文件不存在')
                return JsonResponse({'message': '文件不存在', 'code': 400})
            else:
                file_path = file_object.file_path
                waveform_figure = plot_raw_data_waveform(file_path)

            return JsonResponse({'figure_Base64': waveform_figure})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})


# 用户登录
def login(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        username = data.get('username')
        password = data.get('password')
        # role = data.get('role')

        # print('username: ', username)
        # print('password: ', password)
        # print('role: ', role)

        user = User.objects.filter(username=username).first()
        if not user:
            return JsonResponse({'message': '用户不存在', 'code': 404})
        # 获取用户角色
        user_groups = user.groups.all()

        # 根据用户角色，返回相关的权限信息
        admin_group = Group.objects.filter(name='admin').first()
        superuser_group = Group.objects.filter(name='superuser').first()

        if user_groups.filter(id=admin_group.id).exists():
            role = 'admin'
        elif user_groups.filter(id=superuser_group.id).exists():
            role = 'superuser'
        else:
            role = 'user'
        # print(user_groups)
        # 检查用户角色
        # specific_group_name = role
        # user_is_in_specific_group = any(group.name == specific_group_name for group in user_groups)
        # if not user_is_in_specific_group:
        #     print('不存在该角色的用户')
        #     return JsonResponse({'message': '当前用户没有该角色权限', 'code': 403})
        # 检查登录密码
        if authenticate(request, username=username, password=password):
            payload = {
                'user_id': user.id,
                'username': user.username,
                'exp': datetime.utcnow() + timedelta(hours=24),  # 设置过期时间为24小时后
                'iat': datetime.utcnow(),  # 签发时间
            }
            token = jwt.encode(payload, settings.SECRET_KEY, algorithm='HS256')
            print(f'token: {type(token)}')
            print('登陆成功')
            # 返回JWT以及用户的角色权限信息给客户端
            return JsonResponse({
                'token': token,
                'message': '登陆成功',
                'code': 200,
                'role': role,
            })

        else:
            print('密码错误')
            return JsonResponse({'message': '密码错误', 'code': 405})


# 从http请求中提取token
def extract_jwt_from_request(request) -> str:
    """
    从HTTP请求中提取JWT。
    规定JWT通过Authorization头部以"Bearer <token>"的形式发送。
    """
    auth_header = request.META.get('HTTP_AUTHORIZATION', '')
    prefix = 'Bearer '
    if auth_header.startswith(prefix):
        return auth_header[len(prefix):]
    return ''


# token验证
def verify_jwt(token: str, secret_key: str) -> dict:
    """
    验证JWT并返回其payload。
    如果JWT无效（如签名不匹配、已过期等），则抛出异常。
    """
    try:
        # 设置JWT的验证选项，包括验证过期时间
        options = {
            'verify_signature': True,
            'verify_exp': True,
            'verify_nbf': True,
            'verify_iat': True,
            'verify_aud': False,
            'verify_iss': False,
            'require_exp': True,
            'require_iat': True,
            'require_nbf': False,
            'algorithms': ['HS256'],  # 使用的算法
            'leeway': 0  # 允许的时间误差（秒）
        }

        # 解析JWT
        payload = jwt.decode(token, secret_key, **options)
        return payload
    except jwt.ExpiredSignatureError:
        # 处理JWT过期的情况
        raise Exception('JWT已过期')
    except jwt.InvalidTokenError:
        # 处理JWT无效的情况（如签名不匹配）
        raise Exception('无效的JWT')


# 管理员添加用户操作
def add_user(request):
    if request.method == 'POST':
        data = json.loads(request.body)

        password = data.get('password')
        email = data.get('email')
        jobNumber = data.get('jobNumber')
        role = data.get('role')
        username = data.get('username')

        token = extract_jwt_from_request(request)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            admin_name = payload.get('username')
            if not is_administrator(admin_name):
                return JsonResponse({'message': '没有权限添加用户', 'code': 400})

            if User.objects.filter(jobNumber=jobNumber).exists():
                return JsonResponse({'message': '工号已经存在', 'code': 400})
            elif User.objects.filter(username=username).exists():
                return JsonResponse({'message': '用户名已经存在', 'code': 400})
            else:
                user = User.objects.create_user(username=username, password=password, email=email, jobNumber=jobNumber)
                user_group = Group.objects.get(name='user')
                user.groups.add(user_group)
                if role == 'admin':
                    # 为用户添加管理员权限
                    admin_group = Group.objects.get(name='admin')
                    user.groups.add(admin_group)
                if role == 'superuser':
                    # 为用户添加系统用户权限
                    superuser_group = Group.objects.get(name='superuser')
                    user.groups.add(superuser_group)

                user.save()

                return JsonResponse({'message': 'add user success', 'code': 200})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})


# 管理员删除用户操作
def delete_user(request):
    if request.method == 'GET':
        jobNumber = request.GET.get('jobNumber')
        user = User.objects.get(jobNumber=jobNumber)
        token = extract_jwt_from_request(request)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload.get('username')
            if not is_administrator(username):
                return JsonResponse({'message': '没有权限删除用户', 'code': 400})

            if user:
                user.delete()
                return JsonResponse({'message': 'user deleted success', 'code': 200})
            else:
                return JsonResponse({'message': 'user not found', 'code': 404})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})


# 发送验证码到邮箱
def send_email_captcha(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        username = request.POST.get('username')

        user = User.objects.filter(username=username).first()

        if not user or user.email != email:
            return JsonResponse({'code': 400, 'message': '用户和邮箱不匹配'})
        # 生成验证码
        captcha = ''.join(random.sample(string.digits, k=4))
        # 保存验证码
        models.CaptchaModel.objects.update_or_create(email=email, defaults={'captcha': captcha})
        send_mail(subject="车轮状态分析与健康评估平台验证码",
                  message=f"您的验证码为{captcha}，请勿泄露给他人，并注意使用时限", recipient_list=[email],
                  from_email=None)
        return JsonResponse({"code": 200, "message": "captcha has been send to the email"})


# 验证码校验
def check_captcha(request):
    if request.method == 'POST':
        captcha = request.POST.get('captcha')
        email = request.POST.get('email')

        captcha = models.CaptchaModel.objects.get(email=email, captcha=captcha)
        if not captcha:
            return JsonResponse({'code': 400, 'message': '验证码与邮箱不匹配'})
        else:
            captcha.delete()
            return JsonResponse({'code': 200, 'message': '验证码正确'})


# 用户修改密码
def user_reset_password(request):
    if request.method == 'POST':
        password = request.POST.get('password')
        email = request.POST.get('email')
        # print('password: ', password)
        # print('email: ', email)
        user = User.objects.filter(email=email).first()
        if not user:
            return JsonResponse({"code": 400, 'message': '该用户不存在'})
        else:
            user.set_password(password)
            user.save()
            return JsonResponse({"code": 200, "message": "用户密码修改成功"})


# 验证用户验证码
def authenticate_register(request):
    form_data = json.loads(request.body)
    email = form_data['email']
    # username = form_data['username']
    captcha = form_data['captcha']

    exists = User.objects.filter(email=email).exists()
    if exists:
        return 'email exists'
    else:
        captcha_model = CaptchaModel.objects.filter(email=email, captcha=captcha).first()
        if not captcha_model:
            return 'captcha does not match the email'
        else:
            captcha_model.delete()
            return 'auth success'


# 使用邮箱注册用户
def register(request):
    if request.method == 'POST':
        auth = authenticate_register(request)
        # form表单需包括用户名、邮箱、验证码、密码
        form_data = json.loads(request.body)
        if auth == 'auth success':
            email = form_data.get('email')
            password = form_data.get('password')
            username = form_data.get('username')
            User.objects.create_user(email=email, username=username, password=password)
            return JsonResponse({'code': 200, 'message': 'register success'})
        else:
            print('auth error: ', auth)
            return JsonResponse({'code': 400, 'message': auth})


# 验证用户是否为管理员
def is_administrator(username):
    user = User.objects.get(username=username)
    admin_group = Group.objects.filter(name='admin').first()

    if admin_group in user.groups.all():
        return True
    else:
        return False


# 更新反馈状态
def update_feedback_status(request):
    if request.method == 'POST':
        token = extract_jwt_from_request(request)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})
        try:
            feedback_id = request.POST.get('feedbackId')
            new_status = request.POST.get('newStatus')

            print(f"feedback_id: {feedback_id}, new_status: {new_status}")

            # 更新反馈状态
            feedback = models.Feedback.objects.get(id=feedback_id)
            feedback.status = new_status
            feedback.save()

            return JsonResponse({'message': 'success', 'code': 200})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'error', 'code': 400})


# 返回数据集的列名
def get_dataset_column_names(request):
    if request.method == 'GET':
        dataset_id = request.GET.get('datasetId')
        token = extract_jwt_from_request(request)

        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})

        try:
            dataset = models.SavedDatasetsFromUser.objects.get(id=dataset_id)
            dataset_filepath = dataset.file_path
            # 如果文件不是以.csv或者.xlsx结尾则返回文件类型不支持错误
            if not dataset_filepath.endswith('.csv') and not dataset_filepath.endswith('.xlsx'):
                return JsonResponse({'message': '筛选功能暂不支持该文件类型', 'code': 400})
            df = pd.read_csv(dataset_filepath)
            column_names = df.columns.tolist()
            return JsonResponse({'columnNames': column_names, 'code': 200})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 400})


# 根据用户选择的列名在原数据基础上生成新的数据集
def generate_dataset_with_selected_columns(request):
    if request.method == 'POST':
        dataset_id = request.POST.get('datasetId')
        dataset_name = request.POST.get('datasetName')
        description = request.POST.get('description')
        selected_columns = json.loads(request.POST.get('selectedColumns'))

        token = extract_jwt_from_request(request)

        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})
        # 根据用户选择的列名重新生成数据集
        try:
            dataset = models.SavedDatasetsFromUser.objects.get(id=dataset_id)
            dataset_filepath = dataset.file_path
            df = pd.read_csv(dataset_filepath)
            # 根据用户选择的列名重新生成数据集
            new_df = df[selected_columns]
            # 在原来的文件路径上加上时间戳，再将数据集保存到该路径下
            dataset_filepath = os.path.join(os.path.dirname(dataset_filepath),
                                            f"{os.path.basename(dataset_filepath).split('.')[0]}_{int(time.time())}.csv")
            new_df.to_csv(dataset_filepath, index=False)
            user = User.objects.get(username=payload['username'])
            if models.SavedDatasetsFromUser.objects.filter(dataset_name=dataset_name).exists():
                return JsonResponse({'message': '同名数据集已存在', 'code': 400})
            # 将新的数据集保存到数据库中
            models.SavedDatasetsFromUser.objects.create(dataset_name=dataset_name, file_path=dataset_filepath,
                                                        description=description, owner=user,
                                                        file_type=dataset.file_type,
                                                        multiple_sensors=dataset.multiple_sensors)

            return JsonResponse({'message': 'success', 'code': 200})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 400})


# 将用户指定的多个数据集合并为一个数据集
def concat_dataset(request):
    if request.method == 'POST':
        dataset_ids = json.loads(request.POST.get('datasetIds'))
        dataset_name = request.POST.get('datasetName')
        description = request.POST.get('description')
        # is_public = request.POST.get('isPublic')
        token = extract_jwt_from_request(request)

        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})

        try:
            # 验证所有的数据集是否是相同的传感器数量
            multiple_sensors = models.SavedDatasetsFromUser.objects.get(id=dataset_ids[0]).multiple_sensors
            file_type = models.SavedDatasetsFromUser.objects.get(id=dataset_ids[0]).file_type
            if not all(models.SavedDatasetsFromUser.objects.get(id=dataset_id).multiple_sensors == multiple_sensors for
                       dataset_id in dataset_ids):
                return JsonResponse({'message': '数据集传感器数量不一致', 'code': 400})
            # 检查所有的数据集是否都存在
            if not all(
                    models.SavedDatasetsFromUser.objects.filter(id=dataset_id).exists() for dataset_id in dataset_ids):
                return JsonResponse({'message': '数据集不存在', 'code': 400})
            path_list = [models.SavedDatasetsFromUser.objects.get(id=dataset_id).file_path for dataset_id in
                         dataset_ids]
            df_list = [pd.read_csv(file_path) for file_path in path_list]

            # 验证所有的数据集是否包含相同的列名
            if not all(df.columns.tolist() == df_list[0].columns.tolist() for df in df_list):
                return JsonResponse({'message': '数据集列名不一致', 'code': 400})

            # 将多个数据集合并为一个数据集
            new_df = pd.concat(df_list, ignore_index=True)
            # 在原来的文件路径上加上时间戳，再将数据集保存到该路径下
            dataset_filepath = os.path.join(os.path.dirname(path_list[0]),
                                            f"{os.path.basename(path_list[0]).split('.')[0]}_{int(time.time())}.csv")
            new_df.to_csv(dataset_filepath, index=False)
            user = User.objects.get(username=payload['username'])
            # 如果有重名的数据集则返回错误
            if models.SavedDatasetsFromUser.objects.filter(dataset_name=dataset_name).exists():
                return JsonResponse({'message': '同名数据集已存在', 'code': 400})
            models.SavedDatasetsFromUser.objects.create(dataset_name=dataset_name, file_path=dataset_filepath,
                                                        description=description, owner=user, file_type=file_type,
                                                        multiple_sensors=multiple_sensors)
            return JsonResponse({'message': 'success', 'code': 200})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 405})


# 根据关键字搜索数据集
def search_dataset(request):
    if request.method == 'GET':
        keywords = request.GET.get('keywords')
        token = extract_jwt_from_request(request)

        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})

        username = payload['username']
        user = User.objects.get(username=username)

        # 根据关键字，查询数据库savedDatasetFromUser中文件名包含关键字的文件
        datasets = models.SavedDatasetsFromUser.objects.filter(dataset_name__icontains=keywords, owner=user)

        posts = [
            {
                'id': dataset.id,
                'dataset_name': dataset.dataset_name,
                'description': dataset.description,
                'filepath': dataset.file_path,
                'multiple_sensors': dataset.multiple_sensors,
                # 'publicity': dataset.publicity,
                'owner': dataset.owner.username,
            } for dataset in datasets
        ]

        return JsonResponse({'message': 'success', 'code': 200, 'data': posts})


def generate_output(request):
    if request.method == 'POST':
        token = extract_jwt_from_request(request)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': 'token invalid', 'code': 401})

        try:
            results_to_generate_output = json.loads(request.POST.get('resultsToGenerateOutput'))
            included_modules = results_to_generate_output['includedModules']
            outline = results_to_generate_output['outline']

            # 注册字体格式宋体
            pdfmetrics.registerFont(TTFont('Song', 'SourceHanSansCN-Regular.ttf'))

            # 创建一个BytesIO对象来存储PDF内容
            buffer = BytesIO()

            # 设置文档模板
            p = SimpleDocTemplate(buffer, pagesize=A4)

            # 创建一个列表来存储PDF内容
            content_list = []
            # 定义pdf段落样式
            styles = getSampleStyleSheet()
            styleN = styles['Normal']
            styleCentered = ParagraphStyle(
                name='centered',
                parent=styleN,
                alignment=1,  # 居中对齐
                leftIndent=2 * styleN.fontSize,  # 左侧缩进两个字符宽度
                rightIndent=2 * styleN.fontSize,  # 右侧缩进两个字符宽度
                fontName='Song',
                spaceBefore=22,
                spaceAfter=6
            )
            styleLeftAligned = ParagraphStyle(
                name='leftaligned',
                parent=styleN,
                alignment=0,  # 左对齐
                leftIndent=0,
                rightIndent=0,
                fontName='Song',
                spaceBefore=12,
                spaceAfter=18
            )

            # 将描述文本添加到内容列表中
            description_paragraph = Paragraph(outline, styleLeftAligned)
            content_list.append(description_paragraph)

            # 计数器用于控制每页的内容数量
            content_counter = 0

            for module in included_modules:
                text = results_to_generate_output.get(module).get('text', '')
                image_base64 = results_to_generate_output.get(module).get('imageBase64', '')

                print(f'text for module {module}: ', text)
                print(f'image_base64 for module {module}: ', type(image_base64) if image_base64 else 'None')

                # 将文本添加到内容列表中
                para = Paragraph(text, styleCentered)
                content_list.append(para)

                # 添加图片
                if image_base64:
                    try:
                        img_data = base64.b64decode(image_base64)
                        image_buffer = BytesIO(img_data)

                        img = Image(image_buffer)
                        width, height = img.drawWidth, img.drawHeight
                        wd_ratio = width / height
                        img.drawWidth = 7 * inch if width > 7 * inch else width
                        height_keep_scale = img.drawWidth / wd_ratio
                        img.drawHeight = height_keep_scale if height_keep_scale <= 5 * inch else 5 * inch
                        img.hAlign = 'CENTER'  # 图片水平居中

                        # 将图片添加到故事列表中
                        content_list.append(img)
                    except (TypeError, ValueError) as e:
                        print(f"Error decoding image for module {module}: {str(e)}")
                        continue  # 或者可以选择返回错误响应

                # 增加内容计数器
                content_counter += 1

                # 如果内容计数器达到2，则强制分页
                if content_counter == 2:
                    content_list.append(PageBreak())
                    content_counter = 0

            p.build(content_list)

            # 获取PDF内容
            pdf_content = buffer.getvalue()
            buffer.close()

            # 返回PDF响应
            response = HttpResponse(pdf_content, content_type='application/pdf')
            response['Content-Disposition'] = 'attachment; filename="report.pdf"'
            return response

        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 405})


def resize_image_keep_aspect_ratio(img, max_width=None, max_height=None):
    """
    调整图片大小以适应给定的最大宽度或高度，同时保持图片的宽高比不变。

    :param img: PIL image object
    :param max_width: 最大宽度
    :param max_height: 最大高度
    :return: 调整后的图片对象
    """

    width_percent = (max_width / float(img.size[0]))
    height_percent = (max_height / float(img.size[1]))

    if max_width and max_height:
        percent = min(width_percent, height_percent)
    elif max_width:
        percent = width_percent
    elif max_height:
        percent = height_percent
    else:
        raise ValueError("至少需要指定一个最大宽度或最大高度")

    new_size = (
        int((float(img.size[0]) * float(percent))),
        int((float(img.size[1]) * float(percent)))
    )

    img = img.resize(new_size)
    return img


# 用户建立一个新的树形结构，表示一个新的部件类型
# 树形结构中，每个节点都是一个传感器或模块，每个传感器或模块都有自己的属性，如传感器类型、模块名称、模块描述等
def create_component_tree(request):
    if request.method == "POST":
        token = extract_jwt_from_request(request)

        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload['username']
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 401})
        try:
            # 获取用户的表单数据
            user = User.objects.get(username=username)
            tree_name = request.POST.get('treeName')
            # 如果树名已经存在，则返回错误
            if models.ComponentTree.objects.filter(tree_name=tree_name).exists():
                return JsonResponse({'message': '同名树形结构已存在', 'code': 400})
            # 创建树形结构
            tree = models.ComponentTree.objects.create(tree_name=tree_name)
            # 创建根节点，根节点的标签和值都是树名
            root_node = models.ComponentNode.objects.create(
                component_tree=tree,
                value=tree_name,
                label=tree_name,
                node_level=0,
                parent=None
            )

            return JsonResponse({'message': 'Tree created successfully', 'code': 200})

        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 405})


# 向已有的树形结构中添加节点（添加子类型）
def add_node_to_tree(request):
    if request.method == "POST":
        token = extract_jwt_from_request(request)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload['username']
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 401})
        try:
            user = User.objects.get(username=username)
            tree_name = request.POST.get('treeName')
            tree = models.ComponentTree.objects.get(tree_name=tree_name)
            if not tree:
                return JsonResponse({'message': 'Tree does not exist', 'code': 400})
            # 父节点的value
            node_value = request.POST.get('parentNodeValue')
            # 根据树名和节点值获取父节点
            node = models.ComponentNode.objects.filter(component_tree=tree, value=node_value).first()
            # 从前端获取子节点标签，并将子节点添加到父节点，子节点的value为自动生成
            try:

                new_node_label = '未定义节点'  # 默认节点标签为未定义节点
                new_node = node.add_child(new_node_label)

                new_node_to_response = {
                    'value': new_node.value,
                    'label': new_node.label,
                    'disabled': new_node.disabled,
                    'isModel': new_node.is_model,
                    'modelId': new_node.model.id if new_node.model else '-1',
                    'isPublished': new_node.is_published,
                    'children': []
                }
                return JsonResponse({'message': 'Node added successfully', 'code': 200, 'node': new_node_to_response})
            except Exception as e:
                print(str(e))
                return JsonResponse({'message': str(e), 'code': 405})

        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 400})


# 删除节点
def delete_node(request):
    if request.method == "POST":
        token = extract_jwt_from_request(request)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload['username']
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 401})
        try:

            tree_name = request.POST.get('treeName')
            tree = models.ComponentTree.objects.get(tree_name=tree_name)
            node_value = request.POST.get('nodeValue')
            # 根据树名和节点值获取节点，并删除节点
            node = models.ComponentNode.objects.filter(component_tree=tree, value=node_value).first()
            # print(f"node: {node.value}")
            if not node:
                return JsonResponse({'message': 'Node does not exist', 'code': 400})
            node.delete()
            # 如果model不为None，则删除该节点对应的模型
            if node.model:
                node.model.delete()
            # 如果节点值与树名相同，则删除树
            if node_value == tree_name:
                tree.delete()
            return JsonResponse({'message': 'Node deleted successfully', 'code': 200})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 400})


# 向树形结构中未禁用的节点下保存模型
def save_model_to_tree(request):
    if request.method == "POST":
        token = extract_jwt_from_request(request)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload['username']
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 401})
        try:
            user = User.objects.get(username=username)
            tree_name = request.POST.get('treeName')
            tree = models.ComponentTree.objects.get(tree_name=tree_name)
            # 根据value获取模型实际保存到的类型节点
            node_value = request.POST.get('nodeValue')
            node = models.ComponentNode.objects.get(component_tree=tree, value=node_value)
            if node:
                # 只能向禁用的类型节点中保存模型
                if not node.disabled:
                    return JsonResponse({'message': 'can not add model to this node', 'code': 400})
            else:
                return JsonResponse({'message': 'node not found', 'code': 404})
            # 获取前端提交的模型参数
            model_name = request.POST.get('modelName')
            model_description = request.POST.get('modelDescription')

            # 保存模型
            models.UserModel.objects.create(user=user, model_name=model_name, model_description=model_description,
                                            component_tree=tree, parent_node=node)
            return JsonResponse({'message': 'Model saved successfully', 'code': 200})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 400})


# 根据树名由数据库中存储数据，构造树形结构，以返回给前端进行组件树形分类渲染
def build_component_tree(tree_name):
    # 查询根节点
    root_node = models.ComponentNode.objects.get(component_tree__tree_name=tree_name, parent=None)

    # 递归构建树形结构
    def build_tree(node):
        children = []
        for child in node.children.all():
            child_data = {
                'value': child.value,
                'label': child.label,
                'disabled': child.disabled,
                'isModel': child.is_model,
                'modelId': child.model.id if child.model else '-1',
                'isPublished': child.is_published,
                'children': []
            }
            # 保存的模型挂载到结构树中时，未被禁用且是模型
            if child.disabled or not child.is_model:
                child_data['children'] = build_tree(child)
            children.append(child_data)
        return children

    tree_data = {
        'value': root_node.value,
        'label': root_node.label,
        'disabled': root_node.disabled,
        'isModel': root_node.is_model,
        'modelId': '-1',  # 负数表示没有模型
        'isPublished': root_node.is_published,
        'children': build_tree(root_node)
    }

    return tree_data


# 获取树形结构
def get_component_trees(request):
    if request.method == "GET":
        token = extract_jwt_from_request(request)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload['username']
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 401})
        trees_list = models.ComponentTree.objects.all()

        # 根据列表中的树名获取树形结构
        trees = [build_component_tree(tree.tree_name) for tree in trees_list]

        return JsonResponse({'message': 'Trees retrieved successfully', 'code': 200, 'trees': trees})


# 修改树节点
def edit_node_name(request):
    if request.method == "POST":
        token = extract_jwt_from_request(request)
        try:
            payload = verify_jwt(token, settings.SECRET_KEY)
            username = payload['username']
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 401})
        try:
            tree_name = request.POST.get('treeName')
            tree = models.ComponentTree.objects.get(tree_name=tree_name)
            node_value = request.POST.get('nodeValue')
            new_node_name = request.POST.get('newNodeName')
            node_type = request.POST.get('nodeType')
            # 判断是否为叶子结点，叶子结点不允许再修改，而只能添加模型
            is_disabled = True if node_type == 'node' else False

            print(f"根节点名称：{tree_name}")
            # 如果是根节点，则不允许有其他根节点与其名称相同
            # if node_value == tree_name:
            #     if len(models.ComponentNode.objects.filter(label=new_node_name, node_level=0)) > 1:
            #         return JsonResponse({'message': '节点已存在', 'code': 400})
            node = models.ComponentNode.objects.get(component_tree=tree, value=node_value)

            # 同一级的节点名称不允许重复
            if len(models.ComponentNode.objects.filter(component_tree=tree, label=new_node_name,
                                                       node_level=node.node_level)) > 1:
                return JsonResponse({'message': '同名节点已存在', 'code': 400})
            node.label = new_node_name
            node.disabled = is_disabled
            node.save()
            return JsonResponse({'message': '修改成功', 'code': 200})
        except Exception as e:
            print(str(e))
            return JsonResponse({'message': str(e), 'code': 400})
