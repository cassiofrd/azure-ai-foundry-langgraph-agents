# Sprint 3A — Supervisor multiagente

Implementa roteamento para `inventory`, `supplier`, `time` e `general`.

## Aplicação

Copie os arquivos desta pasta sobre as mesmas pastas do projeto.

## Validação

```powershell
pytest -q
python -m scripts.upload_documents
python -m apps.supervisor.main
```

O upload precisa ser executado para indexar o novo documento de fornecedor do M10.

## Testes manuais sugeridos

- `Qual é a política de estoque do parafuso M10?`
- `Quem fornece o parafuso M10 e qual é o prazo?`
- `Que horas são em UTC?`
- `Explique o que é RAG.`
