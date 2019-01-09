# 快速开始



## 0. 安装Python和相关依赖

* `Python` - 推荐python3.6及以上
* `pip` - 推荐18.1及以上

* 其他：见文件`requirements.txt`，内容如下，安装请见后文：

```python
Django
wooey
django-sslify
django-storages
django-celery
django-autoslug
boto
waitress
psycopg2
collectfast
honcho
```



## 1. 运行使用

### * 创建虚拟环境（非必须，推荐）

为了防止项目工程的依赖和用户自带的python环境相冲突，例如python版本不一致。建议为工程创建虚拟环境，创建方法见[virtualenv](https://virtualenv.pypa.io/en/latest/userguide/), 这里推荐使用[virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/install.html#basic-installation)，一个更快捷的`virtualenv`的封装来创建python3的虚拟环境。

例如，使用`virtualenvwrapper`创建了名称为`wooey`的python3环境，然后在这个环境下运行本项目。

![](assets/workon.png)



## 2. 下载工程代码

![](assets/download.png)

下载地址：[EDS-lab](https://github.com/liuqidev/EDS-lab)

下载后解压，进入到`EDS-lab`路径下。



## 3. 安装相关依赖

使用`pip`安装相关依赖，[requirements.txt](requirements.txt)。

```python
pip install -r requirements.txt
```



## 4. 运行项目

对于*windows*，在当前路径（即`manage.py`所在的路径）下，**分别启动两个终端**：

![](assets/one-terminal.png)

终端1输入：

```powershell
celery -A MyWooProject worker --pool=eventlet -l info
```

效果入上图所示。



![](assets/another-terminal.png)

终端2输入：

```python
python manage.py runserver 0:8000
```

> 注：端口号可以任意指定。



对于Linux和其他操作系统，请查看[这里](https://wooey.readthedocs.io/en/latest/running_wooey.html#through-two-separate-processes)。

## 5.本地查看项目

通过步骤5就构建了一个本地服务器来运行本工程。

![](assets/run.png)

使用浏览器，输入`http://localhost:8000/`,即可查看工程。



## 6. 运行实验脚本

![](assets/run-script.png)

![](assets/run-script-suss.png)

点击脚本名称，输入相关参数，即可运行。例如上图是创建计算`[0, 100)`以内所有正数和的任务，点击提交，任务会被添加到任务队列中，等待片刻，结果输出到界面，结果产生的文件以列表形式陈列于下方。

运行其他脚本过程类似。

