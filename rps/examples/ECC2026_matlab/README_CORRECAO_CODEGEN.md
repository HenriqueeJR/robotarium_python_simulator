# Correcao da equivalencia MATLAB x Codegen

## Problema encontrado

O vetor de previsoes dos vizinhos deveria ocupar `params(132:317)`. No
entanto, `pack_params_1D.m` usava `base = 132` e depois gravava os dados em
`base + 1` e `base + 2`. Assim, `params(132)` permanecia vazio e todas as
coordenadas seguintes ficavam deslocadas em uma posicao.

A funcao preparada para o Codegen lia corretamente a partir de
`params(132)`. Portanto, o erro estava no empacotador, e nao na leitura do
MEX. Isso explica o diagnostico anterior: cada gradiente era consistente
com seu proprio custo, mas a formulacao MATLAB e a formulacao alimentada
ao MEX recebiam trajetorias de vizinhos diferentes.

## Correcao realizada

Em `pack_params_1D.m`, a base foi alterada de 132 para 131. Como a escrita
usa `base + 1`, a primeira coordenada agora ocupa exatamente `params(132)`.

Tambem foi criado `test_equivalencia_codegen.m`, que verifica:

- a recuperacao exata das previsoes empacotadas;
- a igualdade do custo MATLAB e da fonte preparada para Codegen;
- a igualdade dos gradientes;
- o binario MEX, quando existir para a plataforma atual.

O teste do gradiente original foi atualizado com os parametros exigidos
pela geofence analitica atual.

## Validacao obtida

No MATLAB R2025b, apos a correcao:

```text
Erro do empacotamento dos vizinhos : 0.0000e+00
Erro relativo do custo (fonte)     : 0.0000e+00
Erro relativo do gradiente (fonte) : 0.0000e+00
Resultado da fonte                 : PASSOU
```

O teste independente do gradiente analitico tambem passou:

```text
Erro absoluto do gradiente : 1.571953e-09
Erro relativo do gradiente : 1.072817e-09
```

Para repetir:

```matlab
test_equivalencia_codegen
test_gradiente_dcbf
```

## Observacao sobre os binarios

Os arquivos `wrapper_cost_mex.mexw64` e `wrapper_grad_mex.mexw64` foram
gerados para Windows. Eles nao podem ser carregados no macOS, onde a
extensao esperada e `mexmaca64`.

Como o erro estava no empacotador MATLAB, os binarios Windows existentes
passam a receber os dados na ordem correta. A verificacao final nessa
plataforma deve ser feita executando `test_equivalencia_codegen`.

No macOS, os binarios locais foram gerados com o compilador `Xcode
Clang++`. Foram criados `wrapper_cost_mex.mexmaca64` e
`wrapper_grad_mex.mexmaca64`.

A validacao direta dos binarios obteve erro relativo zero no custo e erro
relativo de `2.2019e-18` no gradiente. O diagnostico completo executado no
primeiro passo da simulacao tambem aprovou custo, gradiente, diferencas
finitas, dinamica, campo vetorial e parametros empacotados.

Se o MEX nativo nao existir, `main_multiagente_dcbf.m` seleciona
automaticamente `wrapper_cost.m` e `wrapper_grad.m`. Essa alternativa usa
o mesmo funcional validado, portanto permite desenvolver e simular no Mac,
mas e mais lenta. Quando os arquivos `mexmaca64` forem gerados, o mesmo
programa passa a seleciona-los automaticamente.

Para regenerar os binarios depois de alterar as fontes:

```matlab
mex -setup C++
gerar_mex_local
```

`gerar_mex_local` compila os dois arquivos e executa automaticamente o
teste de equivalencia antes de aprovar os binarios. Os arquivos
intermediarios gerados em `codegen/` nao sao necessarios para executar o
pacote e, por isso, nao foram incluidos na versao de compartilhamento.
