# Generated by Django 5.0.7 on 2024-12-25 08:48

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("app1", "0025_remove_usermodel_user_usermodel_author_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="componentnode",
            name="is_model",
            field=models.BooleanField(default=False, verbose_name="是否为模型"),
        ),
    ]
