# Simulacao de quatro agentes com MPC/PANOC e DCBF

Este pacote contem a linha de base com quatro robos diferenciais. Cada
agente resolve um NMPC local com PANOC e considera as previsoes dos outros
tres agentes na penalidade DCBF. Ainda nao existem formacao de coalizoes,
prioridade de passagem ou custo global de fluxo.

O ensaio inclui:

- seguimento de um ponto de referencia fixo, com custos de estagio e terminal;
- campo vetorial para orientar o movimento nas vias;
- geofence analitica para delimitar a regiao navegavel;
- DCBF interagente com gradiente analitico no forward/backward pass;
- binarios MEX para macOS e Windows, alem da alternativa em MATLAB;
- registro das distancias entre os seis pares e dos tempos computacionais.

## Como executar

1. Descompacte o ZIP.
2. Abra a pasta descompactada no MATLAB.
3. Execute:

```matlab
executar_simulacao_quatro_agentes
```

Tambem e possivel chamar diretamente:

```matlab
results4 = main_quatro_agentes_dcbf(true, 60, 5);
```

O simulador do Robotarium necessario para a execucao local esta incluido
em `Robotarium_simulator`. A simulacao local comeca diretamente nas poses
iniciais do experimento; o controlador utilizado a partir dessas poses nao
e alterado.

Ao final, o script salva `resultado_quatro_agentes_simulacao.mat`.

## Validacao

Antes de executar o ensaio completo, podem ser repetidos os testes:

```matlab
test_equivalencia_codegen
test_gradiente_dcbf
```

O arquivo `resultado_4ag_codegen_mac.mat` registra o ensaio completo
validado no macOS com os binarios MEX nativos. Nesse ensaio:

```text
Missao concluida             : [1 1 1 1]
Tempos de chegada [s]        : [34.419 22.506 36.432 52.107]
Erros finais [m]             : [0.0590 0.0593 0.0596 0.0589]
Distancia minima global [m]  : 0.2465
Distancia fisica requerida   : 0.2000
Violacoes da distancia       : 0
Violacao maxima da geofence  : 0
Possivel bloqueio seguro     : nao
```

Os MEX de macOS foram validados diretamente. Os binarios Windows foram
mantidos para uso nessa plataforma e devem ser conferidos localmente com
`test_equivalencia_codegen` antes da simulacao.

Caso nao exista um MEX compativel com a plataforma atual, o programa usa
automaticamente `wrapper_cost.m` e `wrapper_grad.m`. Essa alternativa
mantem o mesmo funcional, mas apresenta maior tempo de execucao.

## Arquivos principais

- `executar_simulacao_quatro_agentes.m`: execucao direta e salvamento;
- `main_quatro_agentes_dcbf.m`: configuracao do cenario de quatro agentes;
- `main_multiagente_dcbf.m`: simulacao, mapa, campos e controladores;
- `build_local_dcbf.m`: custo e gradiente na implementacao MATLAB;
- `build_local_dcbf_mexable.m`: mesma formulacao preparada para Codegen;
- `wrapper_cost.m` e `wrapper_grad.m`: interfaces do PANOC/Codegen;
- `PanocSolver.m`: solucionador PANOC;
- `campos_quatro_agentes.mat`: campos vetoriais pre-calculados;
- `Robotarium_simulator`: simulador local e respectiva licenca;
- `README_CORRECAO_CODEGEN.md`: correcao e validacao do empacotamento;
- `README_QUATRO_AGENTES.md`: configuracao e interpretacao do ensaio.

O diretorio intermediario `codegen/` nao faz parte do pacote porque pode
ser recriado a qualquer momento. Para recompilar os MEX depois de alterar
as fontes, configure um compilador C++ no MATLAB e execute:

```matlab
mex -setup C++
gerar_mex_local
```
