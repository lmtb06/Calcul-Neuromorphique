# Simulation de neurones biologiquement plausibles

## Structure des dossiers

- **TP/docs/diagrams/src** contient les fichiers puml des diagrammes
- **TP/src** contient le code source:
    - **TP/src/neuromorphic** contient le code de l'ancienne version.
    - **TP/src/neuromorphic_v2** contient le code de la nouvelle version (work in progress mais on peut déjà observer x200 quasiment en terme de performance pour la simulation de neurone).

## Commandes

### Comparer V1 vs V2

```bash
cd TP
uv run neurone_new # Nouvelle version
uv run neurone_old # Ancienne version
```