FROM python:3.10-slim as builder

RUN echo "Types: deb\nURIs: https://mirrors.tuna.tsinghua.edu.cn/debian\nSuites: bookworm bookworm-updates\nComponents: main contrib non-free\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg\n\nTypes: deb\nURIs: https://mirrors.tuna.tsinghua.edu.cn/debian-security\nSuites: bookworm-security\nComponents: main contrib non-free\nSigned-By: /usr/share/keyrings/debian-archive-keyring.gpg" > /etc/apt/sources.list.d/debian.sources
RUN apt-get update && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple && \
    pip config set global.extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set install.trusted-host "pypi.tuna.tsinghua.edu.cn mirrors.aliyun.com"

RUN pip install --user --no-cache-dir -U marker-pdf uvicorn fastapi python-multipart

FROM python:3.10-slim

COPY --from=builder /root/.local /usr/local
COPY . /app
WORKDIR /app
EXPOSE 8000
CMD ["python", "marker_server.py", "--port", "8000"]
