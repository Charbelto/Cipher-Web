from flask import Flask, render_template, request
import numpy as np
import numpy

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/vigenere', methods=['GET', 'POST'])
def vigenere():
    if request.method == 'POST':
        x = request.form['operation']
        plaintext = request.form['plaintext'].lower()
        key = request.form['key'].lower()

        if x == "encrypt":
            result = encrypt_vigenere(plaintext , key, vigenere_dict, dict2)
        else:
            result = decrypt_vigenere(plaintext , key, vigenere_dict, dict2)

        return render_template('vigenere.html', result=result)

    return render_template('vigenere.html')

@app.route('/hill', methods=['GET', 'POST'])
def hill():
    if request.method == 'POST':
        
        m11 = request.form['m11']
        m12 = request.form['m12'] 
        m13 = request.form['m13']
        m21 = request.form['m21']
        m22 = request.form['m22']
        m23 = request.form['m23'] 
        m31 = request.form['m31']
        m32 = request.form['m32']
        m33 = request.form['m33']

        key_matrix = [[int(m11), int(m12), int(m13)], 
                      [int(m21), int(m22), int(m23)],
                      [int(m31), int(m32), int(m33)]]
                      
        k = key_matrix
        operation = request.form['operation']
        plaintext = request.form['plaintext'].lower()
        #key_matrix = request.form['key_matrix']

        if operation == "encrypt":
            result = encrypt_hill(plaintext, k, alphabet_dict, dict2)
        else:
            result = decrypt_hill(plaintext, k, alphabet_dict, dict2)

        return render_template('hill.html', result=result)
		
    return render_template('hill.html')

@app.route('/monoalphabetic', methods=['GET', 'POST'])  
def monoalphabetic():
    if request.method == 'POST':
        operation = request.form['operation']
        plaintext = request.form['plaintext'].lower()
        key = request.form['key'].lower()

        if operation == "encrypt":
            result = encrypt_monoalphabetic(plaintext, key, monoalpha_dict)
        else:
            result = decrypt_monoalphabetic(plaintext, key, monoalpha_dict)

        return render_template('monoalphabetic.html', result=result)

    return render_template('monoalphabetic.html')

@app.route('/affine', methods=['GET', 'POST'])
def affine():
    if request.method == 'POST':
        operation = request.form['operation']
        plaintext = request.form['plaintext'].lower()
        a = int(request.form['a'])
        b = int(request.form['b'])

        if operation == "encrypt":
            result = encrypt_affine(plaintext, a, b, affine_dict)
        elif operation == "decrypt":
            result = decrypt_affine(plaintext, a, b, affine_dict)
        else:
            result = decrypt_affine2(plaintext, a, b, affine_dict)

        return render_template('affine.html', result=result)

    return render_template('affine.html')

@app.route('/playfair', methods=['GET', 'POST'])
def playfair():
    if request.method == 'POST':
        operation = request.form['operation']
        plaintext = request.form['plaintext'].lower()
        key = request.form['key'].lower()

        if operation == "encrypt":
            # Call the encryption function here and store the result in 'result'
            result = encryptByPlayfairCipher(plaintext, key)
        else:
            # Call the decryption function here and store the result in 'result'
            result = decryptByPlayfairCipher(plaintext, key)

        return render_template('playfair.html', result=result)

    return render_template('playfair.html')
@app.route('/extended_euclid', methods=['GET','POST'])
def extended_euclid():
    if request.method == 'POST':
        a = int(request.form['a'])  # No need to lower, convert to integer
        b = int(request.form['b'])  # No need to lower, convert to integer

        result = extended_euclidean(a, b)

        return render_template('extended_euclid.html', result=result)

    return render_template('extended_euclid.html')


# Vigenere cipher helpers
vigenere_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18,'t': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}

dict2 = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}

# Hill cipher helpers                 
alphabet_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18,'t': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}
dict22 = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h', 8: 'i', 9: 'j', 10: 'k', 11: 'l', 12: 'm', 13: 'n', 14: 'o', 15: 'p', 16: 'q', 17: 'r', 18: 's', 19: 't', 20: 'u', 21: 'v', 22: 'w', 23: 'x', 24: 'y', 25: 'z'}



# Monoalphabetic cipher helpers
monoalpha_dict = {'a': 1, 'b': 1, 'c': 1, 'd': 1, 'e': 1, 'f': 1, 'g': 1, 'h': 1, 'i': 1, 'j': 1,'k': 1, 'l': 1, 'm': 1, 'n': 1, 'o': 1, 'p': 1, 'q': 1, 'r': 1, 's': 1,'t': 1, 'u': 1, 'v': 1, 'w': 1, 'x': 1, 'y': 1, 'z': 1}

# Affine cipher helpers                 
affine_dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7, 'i': 8, 'j': 9,
               'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14, 'p': 15, 'q': 16, 'r': 17, 's': 18,
               't': 19, 'u': 20, 'v': 21, 'w': 22, 'x': 23, 'y': 24, 'z': 25}

# Playfair cipher helpers
def toLowerCase(plain):
    return plain.lower()


def removeSpaces(plain):
    return ''.join(plain.split())


def generateKeyTable(key):
    key = key.replace(" ", "")
    key = key.upper()
    table = []
    for i in range(5):
        table.append([])
        for j in range(5):
            table[i].append('')
    
    letters_used = []
    for char in key:
        if char not in letters_used and char != 'J':
            letters_used.append(char)
    
    alphabet = 'ABCDEFGHIKLMNOPQRSTUVWXYZ'
    for char in alphabet:
        if char not in key and char != 'J':
            letters_used.append(char)
    
    for i in range(5):
        for j in range(5):
            table[i][j] = letters_used[5*i + j]
    
    return table

def search(table, a, b):
    x1 = -1
    y1 = -1
    x2 = -1 
    y2 = -1
    
    for i in range(5):
        for j in range(5):
            if table[i][j] == a:
                x1, y1 = i, j
            elif table[i][j] == b:
                x2, y2 = i, j
    
    return x1, y1, x2, y2

def getCoords(char, table):
  for i in range(5):
    for j in range(5):
      if table[i][j] == char:
        return (i, j)
  
  return (-1, -1)

def mod5(a):
    if a < 0:
        a += 5
    return a % 5


def encrypt(plaintext, table):
    plaintext = plaintext.replace(" ", "")
    plaintext = plaintext.upper()
    ciphertext = ""
    
    for i in range(0, len(plaintext)-1, 2):
        p1 = plaintext[i]
        p2 = plaintext[i+1]
        
        if p1 == p2:
            p2 = 'X'
        
        x1, y1, x2, y2 = search(table, p1, p2)
        
        if x1 == x2:
            ciphertext += table[x1][(y1+1)%5] + table[x2][(y2+1)%5] 
        elif y1 == y2:
            ciphertext += table[(x1+1)%5][y1] + table[(x2+1)%5][y2]
        else:
            ciphertext += table[x1][y2] + table[x2][y1]
            
    if len(plaintext) % 2 == 1:
        ciphertext += plaintext[-1]
        
    return ciphertext


def encryptByPlayfairCipher(str, key):
    ks = len(key)
    key = removeSpaces(toLowerCase(key))
    str = removeSpaces(toLowerCase(str))
    keyT = generateKeyTable(key)
    return encrypt(str, keyT)


def decrypt(ciphertext, table):
    ciphertext = ciphertext.upper()
    plaintext = ""
    
    for i in range(0, len(ciphertext)-1, 2):
        c1 = ciphertext[i]
        c2 = ciphertext[i+1]
        
        x1, y1, x2, y2 = search(table, c1, c2)
        
        if x1 == x2:
            plaintext += table[x1][(y1-1)%5] + table[x2][(y2-1)%5] 
        elif y1 == y2:
            plaintext += table[(x1-1)%5][y1] + table[(x2-1)%5][y2]
        else:
            plaintext += table[x1][y2] + table[x2][y1]
    
    return plaintext

def decryptByPlayfairCipher(ciphertext, key):
    key = removeSpaces(toLowerCase(key))
    keyT = generateKeyTable(key)
    return decrypt(ciphertext, keyT)


# Vigenere cipher functions
def encrypt_vigenere(plaintext, key, vigenere_dict,dict2):
    a = len(plaintext)
    b = len(key)
    if a<b:
        key = key[len(plaintext):]

    print("hi")
    cyp = ""
    cyp1 = 0
    j = 0
    while len(key) != len(plaintext):
        
        key = key + key[j % len(key)]
        j = j + 1
    # print(len(key)==len(plaintext))
    i = 0
    while i < len(plaintext):
        if  not plaintext[i].isalpha():
            cyp = cyp + plaintext[i]
            continue
        # print(alphabet_dict[plaintext[i]])
        # print(alphabet_dict[key[i]])
        else:
            cyp1 = (vigenere_dict[plaintext[i]] + vigenere_dict[key[i]]) % 26
            cyp = cyp + dict2[cyp1]
            i = i + 1
    print("hi1")
    print(cyp)
    return cyp

def decrypt_vigenere(cyphertext , key, vigenere_dict,dict2):
    dec = ""
    dec1 = 0
    j = 0
    while len(key) != len(cyphertext):
        
        key = key + key[j % len(key)]
        j = j + 1
    # print(len(key)==len(cypher_text))
    i = 0
    while i < len(cyphertext):
        if  not cyphertext[i].isalpha():
            cyp = cyp + cyphertext[i]
            continue
        # print(alphabet_dict[cypher_text[i]])
        # print(alphabet_dict[key[i]])
        else:
            dec1 = (vigenere_dict[cyphertext[i]] - vigenere_dict[key[i]]) % 26
            dec = dec + dict2[dec1]
            i = i + 1
    
    return dec

# Hill cipher functions
def check_word(k):
    p = len(k) % 3
    if len(k) % 3 != 0:
        for i in range(p):
            k = k + "x"
        #print(k)
        #check_word(k)
    return k

def encrypt_hill(plaintext, k,  alphabet_dict,dict22):
    plaintext = check_word(plaintext)
    print(plaintext)
    p = [[0],[0],[0]]
    cypher = "" 
    for i in range(0,len(plaintext)-1,3):
        if  not plaintext[i].isalpha():
            cypher = cypher + plaintext[i]
            continue
        else:
            entry1 = alphabet_dict[plaintext[i]]
            entry2 = alphabet_dict[plaintext[i+1]]
            if i < len(plaintext)-2:
                entry3 = alphabet_dict[plaintext[i+2]]
            else:
                entry3 = alphabet_dict[plaintext[i+1]]
            p[0][0] = entry1
            p[1][0] = entry2
            p[2][0] = entry3
            result = np.dot(k, p)
            for j in range(3):
                result[j][0] = result[j][0] % 26
            for j in range(3):
                cypher = cypher + dict2[result[j][0]]
    
    return cypher

# Generate inverse key matrix 

def get_inverse_key(key):
    det = int(np.round(np.linalg.det(key)))
    det_inv = Inverse_mod26(det)
    key_inv = (det_inv * np.linalg.inv(key)) % 26
    return key_inv

def Inverse_mod26(det):
    # Inverse modulo 26
    for i in range(26):
        if (det * i) % 26 == 1:
            return i
    return -1

def decrypt_hill(ciphertext, key, alphabet, dict2):
    ciphertext = check_word(ciphertext)
    key_inv = get_inverse_key(key)
    ciphertext_list = [alphabet[c] for c in ciphertext]
    ciphertext_mat = np.reshape(ciphertext_list, (-1, 3))

    plain_mat = np.dot(key_inv, ciphertext_mat) % len(alphabet)
    plain_text = ""
    for row in plain_mat:
        for val in row:
            plain_text += dict2[int(round(val))]

    return plain_text

# Monoalphabetic cipher functions
def encrypt_monoalphabetic(plaintext, key, monoalpha_dict):
    cypher = ""
    for i, letter in enumerate(monoalpha_dict.keys()):
        
        if  not key[i].isalpha():
            cypher = cypher + key[i]
            continue
        if i < len(key):
            monoalpha_dict[letter] = key[i]
    
    for j in range(len(plaintext)):
        if  not plaintext[j].isalpha():
            cypher = cypher + plaintext[j]
            continue
        else:

            cypher = cypher + monoalpha_dict[plaintext[j]]

    return cypher

def decrypt_monoalphabetic(ciphertext, key, monoalpha_dict):
    """Decrypt using Monoalphabetic cipher"""
    plaintext = ""
    for i, letter in enumerate(monoalpha_dict.keys()):
        if  not key[i].isalpha():
            plaintext = plaintext + key[i]
            continue
        if i < len(key):
            monoalpha_dict[letter] = key[i]
    
    def get_key(k, monoalpha_dict):
        for key, value in monoalpha_dict.items():
            if value == k:
                return key
        return None
    for i in range(len(ciphertext)):
        if  not ciphertext[i].isalpha():
            plaintext = plaintext + ciphertext[i]
            continue
        else:
            m = get_key(ciphertext[i],monoalpha_dict)
            plaintext = plaintext + m

    return plaintext

#find multiplicative inverse
def extended_euclidean(m, a):
        A1, A2, A3 = 1, 0, m
        B1, B2, B3 = 0, 1, a
        
        while True:
            if B3 == 0:
                return  None  
            
            if B3 == 1:
                return B2
            
            Q = A3 // B3
            T1, T2, T3 = A1 - (Q * B1), A2 - (Q * B2), A3 - (Q * B3)
            A1, A2, A3 = B1, B2, B3
            B1, B2, B3 = T1, T2, T3
    
# Affine cipher functions
def encrypt_affine(plaintext, a, b, alphabet):
    """Encrypt using Affine cipher"""
    ciphertext = ""

    for char in plaintext:
        if char.isalpha():
            shift = (a*alphabet[char] + b) % 26
            ciphertext += chr(shift + ord('A'))
        else:
            ciphertext += char

    return ciphertext

def decrypt_affine(ciphertext, a, b, alphabet):
    """Decrypt using Affine cipher"""
    dec = ""
    dec1 = 0
    m = 26
    X = extended_euclidean(m,a)

    for i in range(len(ciphertext)):
        if  not ciphertext[i].isalpha():
            dec = dec + ciphertext[i]
            continue
        else:
            dec1 = X*(alphabet[ciphertext[i]]-b ) % 26
            found_keyy = None
            for keyy1, value in alphabet.items():
                if value == dec1:
                    found_keyy = keyy1
                    break
            dec = dec + found_keyy

    return dec

def decrypt_affine2(ciphertext, a, b, alphabet):
    d = {}
    for elem in ciphertext:
        if elem not in d:
            d[elem] = 1
        else:
            d[elem] += 1

    filtered_keys = [key for key, value in d.items() if value > 1]
    sorted_keys = sorted(filtered_keys, key=d.get, reverse=True)
    letter_frequencies = ['e', 't', 'a', 'o', 'i', 'n', 's', 'h', 'r', 'd', 'l', 'c', 'u', 'm', 'f', 'w', 'y', 'g', 'p', 'b', 'v', 'k', 'j', 'x', 'q', 'z']

    decrypt_mapping = dict(zip(filtered_keys, letter_frequencies))

    decrypted_text = ""
    for elem in ciphertext:
        decrypted_text += decrypt_mapping.get(elem, elem)

    return decrypted_text


def modinv(a, m):
    """Modular multiplicative inverse"""
    for x in range(1, m):
        if (a * x) % m == 1:
            return x
    return None
    
if __name__ == '__main__':
    app.run(debug=True)