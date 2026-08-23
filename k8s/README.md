# Phase 9 — Kubernetes (local `kind` / `minikube`)

Deploys the Phase 6 API image plus an in-cluster pgvector database, with all
config in one k8s Secret. Minimal on purpose: 2 Deployments, 2 Services, 1 Secret
— so pods/deployments/services stay easy to see.

```
qna-api  (Deployment + Service)  ── DATABASE_URL ──▶  qna-postgres (Deployment + Service)
        ▲ envFrom qna-secrets                                 ▲ envFrom qna-secrets
```

## Prerequisites
- A local cluster: `kind` or `minikube`, and `kubectl` pointed at it.
- The API image available to the cluster (see step 2).

## 1. Create the Secret (do NOT commit real keys)
Copy the template and fill in your Azure values:
```bash
cp k8s/secret.example.yaml k8s/secret.yaml   # k8s/secret.yaml is git-ignored
# edit k8s/secret.yaml — set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY
# (and LANGFUSE_* if you want tracing). Then:
kubectl apply -f k8s/secret.yaml
```
Use the template rather than `--from-env-file=.env`: the cluster needs the
in-cluster `DATABASE_URL` (pointing at the `qna-postgres` Service) and the
`POSTGRES_*` values, which your local `.env` does not contain.

## 2. Make the image available to the cluster
The manifest uses `imagePullPolicy: IfNotPresent`, so a locally-built image with
the same tag is used without pulling. Build once, then load into your cluster:
```bash
docker build -t ghcr.io/josephchombom/qna-end-to-end-service:latest .
```
- **kind:**     `kind load docker-image ghcr.io/josephchombom/qna-end-to-end-service:latest`
- **minikube:** `minikube image load ghcr.io/josephchombom/qna-end-to-end-service:latest`

(Alternatively, if the GHCR package is public, skip loading and let the cluster
pull it. If it's private, create an image pull secret and reference it.)

## 3. Deploy
```bash
kubectl apply -f k8s/postgres.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

## 4. Watch it come up
```bash
kubectl get pods,svc
kubectl rollout status deployment/qna-api
kubectl logs deploy/qna-api
```

## 5. Answer a query FROM INSIDE the cluster  (the acceptance criterion)
Run a throwaway curl pod that hits the API's ClusterIP Service by name:
```bash
kubectl run curltest --rm -it --image=curlimages/curl --restart=Never -- \
  curl -s -X POST http://qna-api:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How many PTO days do employees get?"}'
```
Expect a JSON answer with sources. (The first call auto-ingests the corpus, so it
may take a few seconds while it embeds via Azure.)

## Reach it from your laptop (optional)
```bash
kubectl port-forward service/qna-api 8000:8000
# then open http://localhost:8000/  or curl localhost:8000/query
```

## Tear down
```bash
kubectl delete -f k8s/service.yaml -f k8s/deployment.yaml -f k8s/postgres.yaml
kubectl delete secret qna-secrets
```
