import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")
%matplotlib inline

arquivo = 'C:/Users/dshn5/Documents/Cursos/Pós-Graduação/Especialização/13_TCC/Dados/household_power_consumption.txt' # caminho
dados = pd.read_csv(arquivo, sep = ";") # carrega o arquivo com o pandas
dados.head() # visualização das 5 primeiras linhas
dados.shape # verificação das linhas x colunas do dataframe
dados.dtypes # verificação do tipo de cada coluna

del dados['Time']

dados['Date'] = pd.to_datetime(dados['Date'], format = '%d/%m/%Y')
dados['Global_active_power'] = pd.to_numeric(dados['Global_active_power'], errors='coerce')
dados['Global_reactive_power'] = pd.to_numeric(dados['Global_reactive_power'], errors='coerce')
dados['Voltage'] = pd.to_numeric(dados['Voltage'], errors='coerce')
dados['Global_intensity'] = pd.to_numeric(dados['Global_intensity'], errors='coerce')
dados['Sub_metering_1'] = pd.to_numeric(dados['Sub_metering_1'], errors='coerce')
dados['Sub_metering_2'] = pd.to_numeric(dados['Sub_metering_2'], errors='coerce')

anos = [2006, 2007, 2008, 2009, 2010]
x = dados['Date'].dt.year
y = dados.groupby(dados['Date'].dt.year)['Global_active_power'].transform('mean')
plt.plot(x, y)
plt.xticks(anos)
plt.title("Demanda Média por Ano (kW)")
plt.show()

del dados['Date']

dados.head()
dados.dtypes # verificação do tipo de cada coluna
dados.describe()
dados.corr(method = 'pearson') # verificando a correlação das colunas entre elas

dados = dados[dados['Voltage'] > 0] # pela inspeção do dataframe, sabe-se que existem valores negativos nessa coluna
dados = dados[dados['Global_active_power'] != None]
dados.dropna(inplace=True)

# verificação da existência de infinitos
dados['Global_active_power'].sum()
dados['Global_reactive_power'].sum()
dados['Voltage'].sum()
dados['Global_intensity'].sum()
dados['Sub_metering_1'].sum()
dados['Sub_metering_2'].sum()
dados['Sub_metering_3'].sum()

dados.head()

X = dados.iloc[:,4:]
Y = dados['Global_active_power'].values

# Gerando autovalores e autovetores
corr = np.corrcoef(X, rowvar = 0)
autovalores, autovetores = np.linalg.eig(corr) # criando autovalores e autovetores

print(autovalores) # como não tem nenhuma variável próximo de zero, então não há multicolinearidade

plt.hist(dados['Sub_metering_1'])
plt.hist(dados['Sub_metering_2'])
plt.hist(dados['Sub_metering_3'])

# Aplicando padronização
modelo_lr = linear_model.LinearRegression() # cria o objeto
standardization = StandardScaler() # chamando a função
stand_coef_linear_reg = make_pipeline(standardization, modelo_lr) # faz a padronização e cria o modelo
stand_coef_linear_reg.fit(X, Y) # aplica a padronização depois treina X e Y

for coef, var in sorted(zip(map(abs, stand_coef_linear_reg.steps[1][1].coef_), dados.columns[4:]), reverse = True):
    print('%6.3f %s' % (coef, var))

X_treino, X_teste, Y_treino, Y_teste = train_test_split(X, Y, test_size = 0.2, random_state = 42) # dividindo treino/teste

# Cria o modelo
modelo = LinearRegression()

# Treina o modelo
modelo2 = modelo.fit(X_treino, Y_treino)

# Calcula a métrica R2 do modelo
r2_score(Y_teste, modelo2.fit(X_treino, Y_treino).predict(X_teste))

modelo2.coef_ #coeficientes
modelo2.intercept_ #constante

predictions=modelo2.predict(X_teste) # fazendo previsões
print(predictions)

# FAZENDO PREVISÕES
vetor = [dados['Sub_metering_1'].mean(), dados['Sub_metering_2'].mean(), dados['Sub_metering_3'].mean()]

# TRANSFORMANDO A LISTA EM ARRAY
vetor_array = np.array(vetor)
print(vetor_array)

# TRANSFORMANDO O ARRAY EM MATRIZ, POIS O MÉTODO PREDICT REQUER ESTA FORMA DE ENTRADA DE DADOS
vetor_array = vetor_array.reshape(1, -1)
print(modelo2.predict(vetor_array))
print("Valores de Entrada = {}, Predição = {}" .format(vetor_array[0], modelo2.predict(vetor_array)[0]))
