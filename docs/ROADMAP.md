# Roadmap — O que fazer com os achados

## Opção 1: Contribuir para projetos existentes (mais impacto, menos risco)

### PR para llama.cpp / MLX
- **MPSGraphExecutable path**: mostrar que pre-compilar o grafo dá 10-170% de speedup
- **MTLTensor para KV cache**: propor uso de tensor views nativos
- **IOReport adaptive backend**: propor switching GPU↔CPU baseado em thermal pressure
- Impacto: milhões de usuários se aceito. Crédito como contributor.

### Blog post / paper técnico
- "Reverse Engineering Apple's Metal 4 ML Pipeline"
- Publicar no Hacker News, r/LocalLLaMA, r/MachineLearning
- Documentar as 1009 IOReport channels, SME2 no M4, MTL4 ML Pipeline
- Gerar visibilidade → consultoria, contratação, ou funding

## Opção 2: Ferramenta de nicho (diferenciada, viável)

### "powerinfer-mac" — Inference engine power-aware para Apple Silicon
O que ninguém faz:
1. **Hybrid ANE+GPU**: prefill na GPU (batch paralelo), decode no ANE (eficiente)
2. **Adaptive compute**: muda backend baseado em IOReport (thermal, battery, throttle)
3. **Pre-compiled model cache**: serializa MPSGraphExecutable em disco, carrega instantâneo
4. **Power budget mode**: "roda o modelo mas sem esquentar / sem gastar bateria"

Target: desenvolvedores que rodam LLMs em MacBook o dia todo e querem:
- Máxima performance quando plugado
- Mínimo consumo quando na bateria
- Sem throttling em sessões longas

### "silicon-monitor" — Dashboard de hardware para Apple Silicon
As 1009 IOReport channels + CPU P-states + GPU thermals em uma TUI/GUI
- Tipo htop mas para o SoC inteiro (CPU, GPU, ANE, DRAM, DCS, ISP, Display)
- Per-core frequency tracking em tempo real
- Power budget por subsistema
- Target: desenvolvedores, reviewers, power users

## Opção 3: Licenciamento / consultoria

### Vender o conhecimento, não o produto
- Empresas que fazem inference on-device (startups de AI, apps)
- Consultoria: "como otimizar seu modelo para Apple Silicon"
- O toolkit de reverse engineering como base para auditorias de performance
- Target: empresas que precisam de edge inference otimizado

## Recomendação

**Fazer na ordem:**
1. Blog post documentando os achados (grátis, gera visibilidade)
2. Open source o toolkit no GitHub (gera credibilidade)
3. PRs para llama.cpp com o MPSGraphExecutable path (impacto direto)
4. Se tiver tração, construir o "powerinfer-mac" como produto

O ativo mais valioso não é o código — é o **conhecimento de como o M4 funciona por dentro** que ninguém mais tem documentado publicamente.
