from pathlib import Path
from base.utils import initialize_server_keys
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
MEDIA_ROOT = os.path.join(BASE_DIR, "media") 

DEFAULT_FILE_STORAGE = "storages.backends.s3boto3.S3Boto3Storage"
# MEDIA_URL = os.environ.get("SUPABASE_S3_ENDPOINT")

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = "django-insecure-9o8plk=l!k(7ytpb)6pm@30f(!9pu0*26_b$4imq$v#!$de#$#"

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = os.environ.get("DEBUG") or False

ALLOWED_HOSTS = [
    "*",
    "localhost",
    "127.0.0.1",
    "backend-h2o4.onrender.com",
    "c894-2401-4900-1c17-eb29-111-cefb-2859-b750.ngrok-free.app",
]

try:
    SERVER_PRIVATE_KEY, SERVER_PUBLIC_KEY = initialize_server_keys()
except ValueError as e:
    print(e)

# Keep this if you want to ensure the logs directory exists
LOG_FILE_PATH = os.path.join(BASE_DIR, "logs", "server.log")
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

# Application definition

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "base.apps.BaseConfig",
    "rest_framework",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "backend.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "backend.wsgi.application"


# Database
# https://docs.djangoproject.com/en/5.1/ref/settings/#databases

DATABASES_NAME = os.environ.get("DATABASES_NAME")
DATABASES_USER = os.environ.get("DATABASES_USER")
DATABASES_PASSWORD = os.environ.get("DATABASES_PASSWORD")
DATABASES_HOST = os.environ.get("DATABASES_HOST")
DATABASES_PORT = os.environ.get("DATABASES_PORT")

#setup the database with your credential
DATABASES = { 
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': DATABASES_NAME,
        'USER': DATABASES_USER,
        'PASSWORD': DATABASES_PASSWORD,
        'HOST': DATABASES_HOST,
        'PORT': DATABASES_PORT,
    },
}

STORAGES = {
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
    },
    "default": {
        "BACKEND": "storages.backends.s3.S3Storage",
        "OPTIONS": {
            "access_key": os.environ.get("SUPABASE_S3_ACCESS_KEY_ID"),
            "secret_key": os.environ.get("SUPABASE_S3_ACCESS_KEY"),
            "bucket_name": os.environ.get("SUPABASE_S3_BUCKET_NAME"),
            "region_name": os.environ.get("SUPABASE_S3_REGION_NAME"),
            "endpoint_url": os.environ.get("SUPABASE_S3_ENDPOINT"),
        },
    },
}



# Password validation
# https://docs.djangoproject.com/en/5.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.1/topics/i18n/

LANGUAGE_CODE = "en-us"

TIME_ZONE = "UTC"

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.1/howto/static-files/

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "static"]

# Default primary key field type
# https://docs.djangoproject.com/en/5.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"