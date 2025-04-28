from datetime import datetime
from django.contrib.auth.models import AbstractUser
# from django.contrib.auth import get_user_model
from django.db import models
from django.conf import settings


# 用户建立的模型
class SavedModelFromUser(models.Model):
    author = models.ForeignKey(to=settings.AUTH_USER_MODEL, on_delete=models.CASCADE, verbose_name="模型建立者")
    model_name = models.CharField(verbose_name="模型名称", max_length=32, null=False, blank=False, default='unknown')
    model_info = models.JSONField(verbose_name="模型信息", null=False, blank=False)
    model_description = models.TextField(verbose_name="模型描述", null=False, blank=False, default='无')
    # is_publish = models.BooleanField(verbose_name="是否公开", null=False, blank=False, default=False)
    is_published = models.CharField(verbose_name="发布状态", max_length=16, null=False, blank=False, default='未发布')

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['author', 'model_name'], name='unique_author_modelName')
        ]


class UserModel(models.Model):
    author = models.ForeignKey(to=settings.AUTH_USER_MODEL, on_delete=models.CASCADE, verbose_name="模型建立者", null=True, default=None)
    model_name = models.CharField(verbose_name="模型名称", max_length=32, null=False, blank=False, default='unknown')
    model_info = models.JSONField(verbose_name="模型信息（包含模型在前端画布中的渲染参数）", null=False, blank=False, default=None)
    model_description = models.TextField(verbose_name="模型描述", null=False, blank=False, default='无')
    # is_publish = models.BooleanField(verbose_name="是否公开", null=False, blank=False, default=False)
    is_published = models.CharField(verbose_name="发布状态", max_length=16, null=False, blank=False, default='未发布')
    report_path = models.CharField(max_length=255, blank=True, null=True)
    model_config = models.JSONField(verbose_name="模型配置信息（包含模块执行顺序以及参数等）", null=False, blank=False, default=None)
    # component_tree = models.ForeignKey(ComponentTree, on_delete=models.CASCADE)
    # parent_node = models.ForeignKey(ComponentNode, on_delete=models.CASCADE)
    # relationship = models.ForeignKey(to=relation, on_delete=models.CASCADE, verbose_name="")

    def __str__(self):
        return self.model_name


# 用户申请发布模型的申请
class PublishModelsApplication(models.Model):
    applicant = models.ForeignKey(to=settings.AUTH_USER_MODEL, on_delete=models.CASCADE, verbose_name="申请人", related_name='applicant_publish_models_applications')
    # model_name = models.CharField(verbose_name="模型名称", max_length=32, null=False, blank=False, default='unknown')
    model = models.ForeignKey(to=UserModel, on_delete=models.CASCADE, verbose_name="模型")
    create_time = models.DateTimeField(verbose_name="申请时间", null=False, blank=False)
    status = models.CharField(verbose_name="申请状态", max_length=16, null=False, blank=False, default='未处理')
    auditor = models.ForeignKey(to=settings.AUTH_USER_MODEL, on_delete=models.CASCADE, verbose_name="审批人", null=True, default=None, related_name='auditor_publish_models_applications')
    audition_time = models.DateTimeField(verbose_name="审批时间", null=True, blank=True)
    report_path = models.CharField(max_length=255, blank=True, null=True)


# 用户所有的数据，以文件路径的形式保存
class SavedDatasetsFromUser(models.Model):
    owner = models.ForeignKey(to=settings.AUTH_USER_MODEL, on_delete=models.CASCADE, verbose_name="数据所有者")
    dataset_name = models.CharField(verbose_name="数据集名称", max_length=64, null=False, blank=False, default='')
    file_path = models.CharField(verbose_name="数据集文件存放路径", max_length=255, null=False, blank=False, default='')
    description = models.TextField(verbose_name="数据集描述", null=False, blank=False, default='无')
    multiple_sensors = models.BooleanField(verbose_name="是否为多传感器数据", null=False, blank=False, default=False)
    publicity = models.BooleanField(verbose_name="是否公开可见", null=False, blank=False, default=False)
    file_type = models.CharField(verbose_name="文件类型", max_length=16, null=False, blank=False, default='未知类型')
    labels_path = models.CharField(verbose_name="标签文件路径", max_length=255, null=True, blank=True, default=None)
    signal_pick_object = models.CharField(verbose_name="数据采集对象", max_length=16, null=False, blank=False, default='unknown')
    sensor = models.CharField(verbose_name="传感器", max_length=255, null=False, blank=False, default='unknown')
    signal_type = models.IntegerField(verbose_name="信号类型 0-振动信号 1-声信号 2-电信号", null=False, blank=False, default=0)
    data_pick_time_start = models.DateTimeField(verbose_name="数据集选取时间段开始", null=True, blank=True, default=None)
    sample_rate = models.IntegerField(verbose_name="采样率", null=False, blank=False, default=0)
    data_pick_time_end = models.DateTimeField(verbose_name="数据集选取时间段结束", null=True, blank=True, default=None)

    # 联合去重
    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['owner', 'dataset_name'], name='unique_owner_dataset')
        ]




# 用户上传的反馈
class Feedback(models.Model):
    user = models.ForeignKey(to=settings.AUTH_USER_MODEL, on_delete=models.CASCADE, verbose_name="反馈者")
    status = models.CharField(verbose_name="反馈处理状态", max_length=16, null=False, blank=False, default='待处理')
    # model_name = models.CharField(verbose_name="反馈的模型名称", max_length=64, null=False, blank=False, default='unknown')
    model = models.ForeignKey(to=UserModel, on_delete=models.CASCADE, verbose_name="反馈的模型id")
    # data_filename = models.CharField(verbose_name="反馈的数据文件名", max_length=64, null=False, blank=False, default='unknown')
    datafile = models.ForeignKey(to=SavedDatasetsFromUser, on_delete=models.CASCADE, verbose_name="反馈的数据id")
    question = models.TextField(verbose_name="反馈的问题", null=False, blank=False, default='')
    module = models.CharField(verbose_name="反馈的模块", max_length=64, null=False, blank=False, default='unknown')
    create_time = models.DateTimeField(verbose_name="反馈时间", null=False, blank=False, auto_now_add=True)


# 用户私有算法
class PrivateAlgorithmOfUser(models.Model):
    owner = models.ForeignKey(to=settings.AUTH_USER_MODEL, on_delete=models.CASCADE, verbose_name="算法所有者")
    algorithm_name = models.CharField(verbose_name="算法名", max_length=256, null=False, blank=False, default='')
    alias = models.CharField(verbose_name="算法别名", max_length=256, null=False, blank=False, default='')
    statement = models.TextField(verbose_name="算法描述", null=False, blank=False, default='无')
    file_path = models.CharField(verbose_name="算法源文件存放路径", max_length=255, null=False, blank=False, default='')
    algorithm_type = models.CharField(verbose_name="算法类型", max_length=32, null=False, blank=False, default='')
    model_filepath = models.CharField(verbose_name="模型文件存放路径", max_length=255, null=False, blank=False,
                                      default='')

    class Meta:
        constraints = [
            models.UniqueConstraint(fields=['owner', 'algorithm_name'], name='unique_owner_algorithm')
        ]


class CustomUser(AbstractUser):
    jobNumber = models.CharField(verbose_name="工号", max_length=32, null=False, blank=False, unique=True,
                                 default='unknown')


# 用于保存用户邮箱验证码
class CaptchaModel(models.Model):
    email = models.EmailField(verbose_name="邮箱地址", unique=True)
    captcha = models.CharField(max_length=4, verbose_name="验证码")
    create_time = models.DateTimeField(verbose_name="发送时间", auto_now_add=True)


# 将用户模型分类保存，其中不同类别以树形结构的形式保存在数据库中
class ModelType(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name


# class ComponentNode(models.Model):
#     component_tree = models.ForeignKey(ComponentTree, on_delete=models.CASCADE)
#     value = models.CharField(max_length=255, unique=True)  # 使用唯一约束确保每个节点的value是唯一的
#     label = models.CharField(max_length=255)
#     parent_value = models.CharField(max_length=255, null=True, blank=True)  # 父节点的value
#     disabled = models.BooleanField(default=False)
#     node_level = models.IntegerField()
#
#     class Meta:
#         unique_together = ('component_tree', 'value')
#
#     def __str__(self):
#         return self.label
#
#     @property
#     def children(self):
#         """获取子节点"""
#         return ComponentNode.objects.filter(parent_value=self.value)
#
#     def add_child(self, label):
#         """为当前节点添加一个子节点"""
#         if not self.disabled:
#             child_value = f"{self.value}.{ComponentNode.get_next_sibling_number(self.value)}"
#             return ComponentNode.objects.create(
#                 component_tree=self.component_tree,
#                 value=child_value,
#                 label=label,
#                 parent_value=self.value,
#                 node_level=self.node_level + 1
#             )
#         else:
#             raise Exception("Cannot add children to a disabled node.")
#
#     @staticmethod
#     def get_next_sibling_number(parent_value):
#         """获取下一个兄弟节点编号"""
#         max_sibling = ComponentNode.objects.filter(parent_value=parent_value).aggregate(models.Max('value'))[
#             'value__max']
#         if max_sibling:
#             return int(max_sibling.split('.')[-1]) + 1
class ComponentTree(models.Model):
    # model_type = models.ForeignKey(ModelType, on_delete=models.CASCADE)
    tree_name = models.CharField(max_length=255, unique=True)

    def __str__(self):
        # return f"{self.model_type} - {self.tree_name}"
        return f"{self.tree_name}"


class ComponentNode(models.Model):
    component_tree = models.ForeignKey('ComponentTree', on_delete=models.CASCADE)
    value = models.CharField(max_length=255, unique=True)
    label = models.CharField(max_length=255)
    parent = models.ForeignKey('self', null=True, blank=True,
                               related_name='children', on_delete=models.CASCADE, default=None)  # 自引用外键
    disabled = models.BooleanField(default=True)
    node_level = models.IntegerField()
    is_model = models.BooleanField(verbose_name="是否为模型", default=False)
    model = models.ForeignKey(UserModel, on_delete=models.CASCADE, null=True, blank=True, default=None)
    is_published = models.CharField(verbose_name="发布状态", max_length=16, null=False, blank=False, default='未发布')

    class Meta:
        unique_together = ('component_tree', 'value')

    def __str__(self):
        return self.label

    @property
    def children(self):
        """获取子节点"""
        return self.children.all()  # 使用related_name访问子节点

    # 只能向禁用的类型节点中保存模型
    # def add_child(self, label, is_model=False, user_model=None, is_published='未发布'):
    #     """为当前节点添加一个子节点，如果是添加类型则is_model==False,user_model==None，否则添加模型"""
    #     print(f"self.value: {self.value}")
    #     child_value = f"{self.value}.{ComponentNode.get_next_sibling_number(self.value)}"
    #     print(f"child_value: {child_value}")
    #     return ComponentNode.objects.create(
    #         component_tree=self.component_tree,
    #         value=child_value,
    #         label=label,
    #         parent=self,  # 直接关联到当前实例
    #         node_level=self.node_level + 1,
    #         is_model=is_model,  # 当前添加的是否为模型节点
    #         model=user_model,
    #         is_published=is_published  # 当前添加的模型的发布状态
    #     )
    def add_child(self, label, is_model=False, user_model=None, is_published='未发布'):
        """为当前节点添加一个子节点，如果是添加类型则is_model==False,user_model==None，否则添加模型"""
        print(f"self.value: {self.value}")
        child_value = f"{self.value}.{ComponentNode.get_next_sibling_number(self.value, self.node_level + 1)}"
        print(f"child_value: {child_value}")
        return ComponentNode.objects.create(
            component_tree=self.component_tree,
            value=child_value,
            label=label,
            parent=self,  # 直接关联到当前实例作为新增节点的父节点
            node_level=self.node_level + 1,
            is_model=is_model,  # 当前添加的是否为模型节点
            model=user_model,
            is_published=is_published  # 当前添加的模型的发布状态
        )

    @staticmethod
    def get_next_sibling_number(parent_value, node_level):
        """获取下一个兄弟节点编号"""
        # 获取所有符合条件的子节点
        siblings = ComponentNode.objects.filter(
            value__startswith=f"{parent_value}.",
            node_level=node_level
        )
        # 提取节点值的最后一个部分并转换为整数进行比较
        max_sibling_number = 0
        for sibling in siblings:
            parts = sibling.value.split('.')
            last_part = parts[-1]
            try:
                number = int(last_part)
                if number > max_sibling_number:
                    max_sibling_number = number
                print(f"max_sibling_number: {max_sibling_number}")
            except ValueError:
                # 如果无法转换为整数，跳过该节点
                continue

        return max_sibling_number + 1

    def delete(self, *args, **kwargs):
        """删除节点时，删除其子节点"""
        children = self.children.all()
        for child in children:
            if child.model:
                child.model.delete()
        super().delete(*args, **kwargs)


# class RelationOfModelAndTree(models.Model):
#
#     """模型和树之间的关联"""
#     treenode = models.ForeignKey(ComponentNode, on_delete=models.CASCADE)
