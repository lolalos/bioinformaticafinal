import requests

# === CONFIGURA TU TOKEN DE DRUGBANK ===
DRUGBANK_TOKEN = "TU_TOKEN_DRUGBANK_AQUI"  # Regístrate en https://go.drugbank.com

def buscar_en_drugbank(drug_name):
    url = f"https://go.drugbank.com/unearth/q?searcher=drugs&query={drug_name}"
    headers = {"Authorization": f"Token {DRUGBANK_TOKEN}"}
    response = requests.get(url, headers=headers)
    if response.ok:
        return response.json()
    return {"error": "No se pudo acceder a DrugBank"}

def buscar_en_pubchem(drug_name):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drug_name}/JSON"
    response = requests.get(url)
    if response.ok:
        data = response.json()
        compound = data.get("PC_Compounds", [])[0]
        props = {p["urn"]["label"]: p.get("value", {}).get("sval", "") for p in compound.get("props", []) if "urn" in p}
        return props
    return {"error": "No encontrado en PubC"
        "hem"}

def buscar_en_chembl_por_indicacion(indicacion="cancer", limit=5):
    url = f"https://www.ebi.ac.uk/chembl/api/data/indication.json?mesh_heading={indicacion}&limit={limit}"
    response = requests.get(url)
    if response.ok:
        data = response.json()
        resultados = [{
            "drug_chembl_id": i["drug_chembl_id"],
            "mesh_heading": i["mesh_heading"]
        } for i in data.get("indications", [])]
        return resultados
    return {"error": "No se encontraron resultados en ChEMBL"}

# === Función principal de consulta ===
def buscar_medicamento(drug_name):
    print(f"\n🔎 Buscando información sobre: {drug_name}")

    print("\n📘 DrugBank:")
    drugbank_data = buscar_en_drugbank(drug_name)
    print(drugbank_data if "error" in drugbank_data else drugbank_data.get("data", [{}])[0])

    print("\n🧪 PubChem:")
    pubchem_data = buscar_en_pubchem(drug_name)
    for k, v in list(pubchem_data.items())[:10]:  # Muestra solo los primeros 10 atributos
        print(f"{k}: {v}")

    print("\n🧬 ChEMBL (relación con cáncer):")
    chembl_data = buscar_en_chembl_por_indicacion()
    for item in chembl_data:
        print(f"{item['drug_chembl_id']} - {item['mesh_heading']}")

# === EJECUCIÓN ===
if __name__ == "__main__":
    buscar_medicamento("doxorubicin")
