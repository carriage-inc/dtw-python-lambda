FROM --platform=linux/amd64 public.ecr.aws/lambda/python:3.8

COPY requirements.txt ${LAMBDA_TASK_ROOT}
COPY handler.py ${LAMBDA_TASK_ROOT}
COPY mfcc_dtw.py ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt

# FFmpegのバイナリをコピー
COPY ffmpeg-6.0-amd64-static/ffmpeg /usr/local/bin/ffmpeg
COPY ffmpeg-6.0-amd64-static/ffprobe /usr/local/bin/ffprobe

# 実行権限を設定
RUN chmod 755 /usr/local/bin/ffmpeg
RUN chmod 755 /usr/local/bin/ffprobe

ENV NUMBA_CACHE_DIR /tmp/cache1

CMD [ "handler.calc_mfcc_dtw" ]
