from selenium import webdriver
import chromedriver_autoinstaller
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
import json
import time
import re
from colored import Fore, Back, Style


chromedriver_autoinstaller.install()
driver = webdriver.Chrome()
car_data_set = set()

dict_list = []


def get_car_data(url):


    driver.get(url)
    time.sleep(3)
    try:
        button_accept = driver.find_element(By.XPATH,"//button[@id='onetrust-accept-btn-handler']")
        button_accept.click()
    except:
        #Bir şey yapmamızıa gerek yok sadece programın durmasını önler
        bos_islem = 0
    

    try:
        cookie_button = driver.find_element(By.XPATH,"//div[@id='head-tab-equipment-information']")
        cookie_button.click()
        time.sleep(3)
    except:
        print(f"{Fore.rgb(255, 191, 0)}Sayfada donanım kısmı bulunmamaktadır.")
    

    dict = {
        "İlan No": -1,
        "İlan İsmi" : -1,
        "İlan Konumu" : "Yok",
        "İlan Tarihi" : "Yok",
        "Marka" : "Yok",
        "Model" : "Yok",
        "Yıl": -1,
        "Kilometre": -1,
        "Renk" : "Yok",
        "Garanti Durumu":"Yok",
        "Aracın ilk sahibiyim": "Yok",
        "Araç Türü":"Yok",
        "Araç Durumu" : "Yok",
        "Sınıfı" : "Yok",
        "Satıcı" : "Yok",
        "Boya-değişen" : "Yok",
        "Takasa Uygun" : "Yok",
        "Plaka Uyruğu" : "Yok",
        "Kimden" : "Yok",
        "Ortalama Kasko" : -1,
        "Ortalama Trafik Sigortası" : -1,
        "Fiyat" : -1,
        "Vites Tipi" : "Yok",
        "Yakıt Tipi" : "Yok",
        "Kasa Tipi" : "Yok",
        "Çekiş" : "Yok",
        "Motor Hacmi" : -1,
        "Motor Gücü" : -1,
        "Tork" : -1,
        "Silindir Sayısı": -1,
        "Maksimum Güç" : -1,
        "Minimum Güç" : -1,
        "Hızlanma (0-100)": -1,
        "Maksimum Hız": -1,
        "Yakıt Deposu" : -1,
        "Ortalama Yakıt Tüketimi":-1,
        "Şehir İçi Yakıt Tüketimi":-1,
        "Şehir Dışı Yakıt Tüketimi":-1,
        "Uzunluk":-1,
        "Genişlik": -1,
        "Yükseklik":-1,
        "Boş Ağırlığı": -1,
        "Koltuk Sayısı": -1,
        "Bagaj Hacmi":-1,
        "Ön Lastik": "Yok",
        "Aks Aralığı": -1,
        "Donanım Özellikleri": "Yok"
    }



    dict.update({"Link":url})
    #Price fiyatını alma
    try:
        price = str(driver.find_element(By.XPATH,"//div[@data-testid='desktop-information-price']").get_attribute("textContent"))
        price = int( price.strip().replace('TL','').replace('.',''))
        dict.update({"Fiyat" : price})
    except:
        print(f"{Fore.rgb(255, 0, 0)}Price bilgisi bulunamadı!")
    
    #İlan No
    try:
        listing_num = driver.find_element(By.XPATH,"//div[@id='js-hook-copy-text']").get_attribute("textContent").strip()
        dict.update({"İlan No" : listing_num})
    except:
        print(f"{Fore.rgb(255, 191, 0)}İlan Numarası bulunamadı!")
    
    #İlan İsmi
    try:
        listing_name = str(driver.find_element(By.XPATH,"//div[@class='product-name-container']").get_attribute("textContent").strip())
        dict.update({"İlan İsmi" : listing_name})
    except:
        print(f"{Fore.rgb(255, 191, 0)}İlan ismi bulunamadı!")
    
    #İlan Konumu
    try:
        listing_place = str(driver.find_element(By.XPATH,"//span[@class='product-location']").get_attribute("textContent").strip())
        dict.update({"İlan Konumu" : listing_place})
    except:
        print(f"{Fore.rgb(255, 191, 0)}İlan konumu bulunamadı!")
    
    #Satıcı advert-owner-name
    try:
        listing_owner_name  = driver.find_element(By.XPATH,"//div[@class='advert-owner-name']").get_attribute("textContent").strip()
        dict.update({"Satıcı" : listing_owner_name})
    except:
        print(f"{Fore.rgb(255, 191, 0)}Satıcı ismi bulunamadı!")
    #Boya-Değişen
    # Burada hata alıyorum özel bir string olunca sıkıntı çıkarıyor incelemeye alalım.
    
    try:
        listing_owner_boya  = driver.find_element(By.XPATH,"//div[@class='property-value property-highlighted 44']").get_attribute("textContent").strip()
        dict.update({"Boya-değişen" : listing_owner_boya})
    except: 
        print(f"{Fore.rgb(255, 191, 0)}Boya-değişen highlighted tipinde değil")
   

    donanim_liste = []
    
    try: 
        listing_donanim  = driver.find_element(By.XPATH,"//div[@class='equipment-list']")
        listing_donanim  = listing_donanim.find_element(By.XPATH,"//div[@class='equipment-list']")
        listing_div = listing_donanim.find_elements(By.XPATH,"//div[@class='item']")
        
        for item in listing_div:
            item_text = item.get_attribute("textContent").strip()
            donanim_liste.append(item_text) 

        dict.update({"Donanım Özellikleri" : donanim_liste})

    except:
        print(f"{Fore.rgb(255, 191, 0)}Araba sayfasında donanımla ilgili özellikler belirtilmemiş")
    #Sayfanın üstündeki bilgileri çekmeden bahsetmekteyiz.
    
    try:
        property_list = WebDriverWait(driver, 5).until(
                            EC.presence_of_all_elements_located((By.XPATH,"//div[@class='property-item']"))
                )
    except:
        print(f"{Fore.rgb(255, 191, 0)}Veri gereken zamanda yüklenemedi adım atlanıyor")
        property_list = []

    for propert_item in property_list :
        try:
            key = str(propert_item.find_element(By.XPATH,".//div[@class='property-key']").get_attribute("textContent"))
            value = str(propert_item.find_element(By.XPATH,".//div[@class='property-value']").get_attribute("textContent"))
            key = key.strip()
            value = value.strip()

            control_value = dict.get(key, "empty")
            if key == "İlan No" or control_value == "empty":
                continue

            x = re.findall("[0-9]", value)
    
            #Gelen string verilerini integera dönüştürür.
            if x and (key != "Model" and key != "İlan Tarihi") :
                try: 
                    value = float(value.replace("km","").replace("cc","").replace("hp","").replace("lt","").replace("mm","").replace("TL","").replace("rpm","").replace("km/s","").replace("kg","").replace(".","").replace(",",".").strip())
                except:
                    print(f"{Fore.rgb(255, 191, 0)}Veri çeviri listemize uygun değil string olarak kaydedildi!")
            #Bu tip toplamada sorun çıkaran elemanlar çıkartılır.
            control_value = dict.get(key, "empty")
            if key == "İlan No" or control_value == "empty":
                continue

            dict.update({f"{key}" : value})

        except:
            print(f"{Fore.rgb(255, 191, 0)}Bir property kullanıcı tarafından açıklanmamış")

    try:
        car_information_tab = driver.find_element(By.XPATH,"//div[@class='tab-content-car-information']")
        info_list= car_information_tab.find_elements(By.CSS_SELECTOR,"li")
    except:
        print(f"{Fore.rgb(255, 0, 0)}Araçın bilgisinin olduğu alt tab bulunamadı")
        info_list = []

    for info in info_list:

        try:

            span_list = info.find_elements(By.TAG_NAME,"span")
            key = span_list[0].get_attribute("textContent")
            value = span_list[1].get_attribute("textContent")

            control_value = dict.get(key, "empty")
            if control_value == "empty":
                
                continue

            x = re.findall("[0-9]", value)

            

            if x and key != "Ön Lastik":
                try:
                    value = float(value.replace("km","").replace("cc","").replace("hp","").replace("lt","").replace("mm","").replace("TL","").replace("rpm","").replace("km/s","").replace("/s","").replace("kg","").replace(".","").replace("nm","").replace("sn","").replace(",",".").strip())
                except:
                    print(f"{Fore.rgb(255, 191, 0)}Veri çeviri listemize uygun değil string olarak kaydedildi!")

            

            dict.update({f"{key}" : value})

        except:
            print(f"{Fore.rgb(255, 0, 0)}Araça bilgisi işlenirken bir sorun oluştu!")

    

    dict_list.append(dict)





# Rıdvan 
#car_model_list = ["bmw","audi","citroen","fiat","ford","honda","hyundai"]

# Tavlan
#car_model_list = ["mercedes-benz","opel","peugeot","renault","seat","skoda","tofas"]

# Mirza chevrolette sorun oluştu onu düzeltirsin
car_model_list = ["toyota","volkswagen","chevrolet", "dacia", "kia", "nissan", "volvo"]


json_data = []

for car_model in car_model_list:

    with open(f"car_{car_model}_links.json","r") as file:
        json_data = json.load(file)
        file.close()



    index = 0
    start_time = time.time()

    #Önceki aşamalarda toplanmış veri sayılarına bak

    try:
        with open(f"car_data_{car_model}.json","r",encoding="utf-8") as file:
            json_control = json.load(file)
            index = len(json_control) - 1
                
            for i in range(index):
                dict_list.append(json_control[i])
    except:
        print(f"{Fore.rgb(255, 191, 0)}Elimizde düşündüğümüz gibi bir dosya bulunmamaktadır!")

    for url in json_data[index:]:
        index = index + 1 
        
        if url:
            get_car_data(url)    
        finish_time = time.time()

        print(f"{Fore.rgb(0, 255, 0)} {index}. Araba  Süre => {finish_time - start_time}s")
        if(index % 25 == 0):
            with open(f"car_data_{car_model}.json","w",encoding="utf-8") as file:
                json.dump(dict_list,file,ensure_ascii=False, indent=4) 
            print(f"{Fore.rgb(0, 255, 0)}Index {index}'e kadar işlem başarıyla tamamlandı.")

    driver.quit()