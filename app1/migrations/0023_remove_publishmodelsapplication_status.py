# Generated by Django 5.0.7 on 2024-12-24 06:56

from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ("app1", "0022_rename_root_node_usermodel_parent_node_and_more"),
    ]

    operations = [
        migrations.RemoveField(
            model_name="publishmodelsapplication",
            name="status",
        ),
    ]
