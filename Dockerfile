FROM python:3.10-slim as builder

#RUN echo "deb https://mirrors.aliyun.com/debian/ bookworm main contrib non-free non-free-firmware" > /etc/apt/sources.list && \
#    echo "deb https://mirrors.aliyun.com/debian/ bookworm-updates main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
#    echo "deb https://mirrors.aliyun.com/debian/ bookworm-backports main contrib non-free non-free-firmware" >> /etc/apt/sources.list && \
#    echo "deb https://mirrors.aliyun.com/debian-security bookworm-security main contrib non-free non-free-firmware" >> /etc/apt/sources.list

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip config set global.extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set install.trusted-host "pypi.tuna.tsinghua.edu.cn mirrors.aliyun.com"

RUN pip install --user --no-cache-dir marker-pdf uvicorn fastapi python-multipart

FROM python:3.10-slim

COPY --from=builder /root/.local /usr/local
COPY . /app
WORKDIR /app
EXPOSE 8000
CMD ["python", "marker_server.py", "--port", "8000"]
