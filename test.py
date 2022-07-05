letters = " #'%()+,-./:0123456789ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
label_dict = {letters.index(c): c for c in letters}
label_dict[140] = 'blank'

all_labels = list(label_dict.keys())
all_chars = list(label_dict.values())

def label_to_text(label):
    return [label_dict[l] for l in label]

def text_to_label(text):
    return [all_labels[all_chars.index(c)] for c in text]

print(text_to_label("abc"))


