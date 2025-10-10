# CI/CD for Model Deployment

> Automate testing, building, and deployment with GitLab CI.

---

## Recommended Workflow

1. Push code or model version to Git repository  
2. Run **unit & integration tests**  
3. Build Docker image, tag with version  
4. Deploy via **Helm/Kubernetes**  
5. Run **post-deployment tests & health checks**  

---

## Example `.gitlab-ci.yml`

```yaml
stages:
  - test
  - build
  - deploy

variables:
  IMAGE_NAME: registry.gitlab.com/mygroup/mymodel

unit_tests:
  stage: test
  image: python:3.12
  script:
    - pip install -r requirements.txt
    - pytest tests/

build_docker:
  stage: build
  image: docker:24.0
  services:
    - docker:dind
  script:
    - docker build -t $IMAGE_NAME:$CI_COMMIT_SHA .
    - docker push $IMAGE_NAME:$CI_COMMIT_SHA

deploy_helm:
  stage: deploy
  image: alpine/helm:3.12.0
  script:
    - helm upgrade --install mymodel ./charts/mymodel --set image.tag=$CI_COMMIT_SHA
```

---

### Tips & Warnings

!!! tip
     * Use **CI_COMMIT_SHA** or version tags for reproducible deployments
     * Run **tests first** to prevent broken production releases
     * Use **environment variables** for secrets (avoid hardcoding in `.gitlab-ci.yml`)

!!! warning
     * Don’t manually update pods in production — breaks traceability
     * Always validate Helm charts and post-deploy health probes