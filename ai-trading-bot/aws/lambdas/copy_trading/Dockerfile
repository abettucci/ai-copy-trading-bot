FROM public.ecr.aws/lambda/python:3.11

COPY copy_trading/requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir

COPY shared/ ${LAMBDA_TASK_ROOT}/shared/
COPY copy_trading/ ${LAMBDA_TASK_ROOT}/

RUN rm -rf /var/cache/pip/* /tmp/* /var/tmp/* && \
    find ${LAMBDA_TASK_ROOT} -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

CMD ["handler.lambda_handler"]
