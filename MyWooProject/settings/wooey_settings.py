from .django_settings import *
from wooey.version import DJANGO_VERSION, DJ110
from django.utils.translation import ugettext_lazy as _

INSTALLED_APPS += (
    # 'corsheaders',
    'wooey',
)

if DJANGO_VERSION < DJ110:
    MIDDLEWARE_CLASSES = list(MIDDLEWARE_CLASSES)
    MIDDLEWARE_CLASSES.append('MyWooProject.middleware.ProcessExceptionMiddleware')
    MIDDLEWARE_OBJ = MIDDLEWARE_CLASSES
else:
    # Using Django 1.10 +
    MIDDLEWARE = list(MIDDLEWARE)
    MIDDLEWARE.append('MyWooProject.middleware.ProcessExceptionMiddleware')
    MIDDLEWARE_OBJ = MIDDLEWARE

LANGUAGES = [
  ('de', _('German')),
  ('en', _('English')),
  ('fr', _('French')),
  ('ja', _('Japanese')),
  ('nl', _('Dutch')),
  ('zh-hans', _('Simplified Chinese')),
]

NEW_MIDDLEWARE = []
for i in MIDDLEWARE_OBJ:
    NEW_MIDDLEWARE.append(i)
    if i == 'django.contrib.sessions.middleware.SessionMiddleware':
        NEW_MIDDLEWARE.append('django.middleware.locale.LocaleMiddleware')

NEW_MIDDLEWARE.append('MyWooProject.middleware.ProcessExceptionMiddleware')
if DJANGO_VERSION < DJ110:
    MIDDLEWARE_CLASSES = NEW_MIDDLEWARE
else:
    MIDDLEWARE = NEW_MIDDLEWARE

PROJECT_NAME = "MyWooProject"
WOOEY_CELERY_APP_NAME = 'wooey.celery'
WOOEY_CELERY_TASKS = 'wooey.tasks'
WOOEY_SITE_NAME=u'EDS Lab'
WOOEY_SITE_TAG='实验设计与数据分析可视化用户界面'
