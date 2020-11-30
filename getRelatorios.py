import pandas as pd
import re
import unicodecsv as csv

def writeToCSV(nome, data):
    # Data tem que ser lista de listas
    # Nome tem que terminar com .csv
    arquivo = open(nome , "w+", newline = "")
    with arquivo:
        write = csv.writer(arquivo)
        write.writerows(data)


def getRelatorio(sent):
    temp = sent.lower()
    fim = re.search('é o relatório', temp)
    # TODO aprimorar ementa
    ementa = re.search('provido.', temp)
    if fim == None or ementa == None:
        return None
    return sent[ementa.start()+7:fim.start()+14]


def relatorios(cases):
    final = []
    i = 0
    while i < len(cases):
        print(i)
        if getRelatorio(cases.julgado[i]) != None:
            final.append((getRelatorio(cases.julgado[i]), cases.resultado[i]))
        i += 1
    return final


def main():
    import pdb
    csvBase = pd.read_csv("acordaos_full.csv", encoding="ISO-8859-1")
    print("1")
    relat_data = relatorios(csvBase)
    relat_data = pd.DataFrame(relat_data)
    print("2")
    relat_data.to_csv("relatorios_full.csv", header = ["julgado","resultado"])
   # writeToCSV("relatorios.csv", relat_data)



if __name__ == "__main__":
    main()