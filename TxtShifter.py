from fontTools.ttLib import TTFont  # type: ignore #
import json
import random
import string
import argparse

#(MIN, MAX)
NUMBER = (0x30,0x39)
ALPHA_HIGH = (0x41,0x5A)
ALPHA_LOW = (0x61,0x7A)
K_LIST = (0xAC00, 0xD7A3)

class FontConverter:
    def __init__(self, font_path):
        self.font = TTFont(font_path)

        cmp = self.font['cmap']
        bcmap = self.font.getBestCmap()

        self.mapping_list = [(code_point,glyph_name) for code_point, glyph_name in bcmap.items()]
        self.mapping_list.sort()
        print(f"MAPPING LIST MAKED. TEST DATA IS {self.mapping_list[0]}")


        
    def shuffle(self):
        tmp_number_list = []
        tmp_alpha_high_list = []
        tmp_alpha_low_list = []
        tmp_korean_list = []

        for k in self.mapping_list:
            if NUMBER[0] <= k[0] <= NUMBER[1]:
                tmp_number_list.append(k)
            if ALPHA_HIGH[0] <= k[0] <= ALPHA_HIGH[1]:
                tmp_alpha_high_list.append(k)
            if ALPHA_LOW[0] <= k[0] <= ALPHA_LOW[1]:
                tmp_alpha_low_list.append(k)
            if K_LIST[0] <= k[0] <= K_LIST[1]:
                tmp_korean_list.append(k)
        
        random.shuffle(tmp_number_list)

        tmp_alpha_list = tmp_alpha_high_list + tmp_alpha_low_list
        random.shuffle(tmp_alpha_list)
        tmp_alpha_high_list = tmp_alpha_list[:26]
        tmp_alpha_low_list = tmp_alpha_list[26:]

        random.shuffle(tmp_korean_list)


        self.shuffle_list = []
        self.norm_to_shuffled_list = []

        for k in self.mapping_list:
            if NUMBER[0] <= k[0] <= NUMBER[1]:
                ugtuple = (tmp_number_list[k[0]-NUMBER[0]][0],k[1])
            elif ALPHA_HIGH[0] <= k[0] <= ALPHA_HIGH[1]:
                ugtuple = (tmp_alpha_high_list[k[0]-ALPHA_HIGH[0]][0],k[1])
            elif ALPHA_LOW[0] <= k[0] <= ALPHA_LOW[1]:
                ugtuple = (tmp_alpha_low_list[k[0]-ALPHA_LOW[0]][0],k[1])
            elif K_LIST[0] <= k[0] <= K_LIST[1]:
                ugtuple = (tmp_korean_list[k[0]-K_LIST[0]][0],k[1])
            else:
                ugtuple = k

            self.shuffle_list.append(ugtuple)
            self.norm_to_shuffled_list.append((k[0],ugtuple[0]))

        self.shuffle_list.sort()
        self.norm_to_shuffled_list.sort()

        print(self.shuffle_list[0x30:0x35])
        print(self.norm_to_shuffled_list[0x30:0x35])
            
            
        
    def export(self,ext_font_name ,ext_font_path, convert_table_path):
        new_font = self.font
        cmap_table = new_font['cmap'].getcmap(3, 1)

        name_table = new_font['name']
        for record in name_table.names:
            if record.nameID in [1, 3, 4, 6]:  # 중요한 nameID들
                try:
                    record.string = ext_font_name.encode(record.getEncoding())
                except:
                    record.string = ext_font_name.encode('utf-16-be')  # fallback

        for k in self.shuffle_list:
            cmap_table.cmap[k[0]] = k[1]
        
        new_font.save(ext_font_path)
        convert_dict = {hex(orig_cp): hex(shuf_cp) for orig_cp, shuf_cp in self.norm_to_shuffled_list}
        with open(convert_table_path, 'w', encoding='utf-8') as f:
            json.dump(convert_dict, f, indent=2, ensure_ascii=False)


class TextConverter: 
    def __init__(self, convert_table_path):
        with open(convert_table_path, 'r', encoding='utf-8') as f:
            self.convert_dict = json.load(f)
        self.convert_dict = {int(k,16): int(v,16) for k, v in self.convert_dict.items()}

        #print(f"test str1: a -> {ord("a")} -> {self.convert_dict[ord("a")]} -> {chr(self.convert_dict[ord('a')])}")
        #print(f"test str2: 가 -> {ord("가")} -> {self.convert_dict[ord("가")]} -> {chr(self.convert_dict[ord('가')])}")
        

    def convert_text(self, target):
        rtn = ""
        for ch in target:
            rtn += chr(self.convert_dict[ord(ch)])

        return rtn


class RandomImageTagGenerator:
    def __init__(self):
        pass

    def make_random_tag(self, image_path):
        rand_alt = ''.join(random.choices(string.ascii_letters + string.digits, k=16))
        return f'<img src="{image_path}" alt="{rand_alt}">'
    
DEFAULT_TARGET_FONT = "NaneumGothic.ttf"
DEFAULT_TARGET_EXT_FONT = "example.ttf"
DEFAULT_TARGET_FONT_NAME = "example"
DEFAULT_CONVERT_DICT_NAME = "example.json"

DEFAULT_CONVERTED_STRING_NAME = "이건 테스트를 위해 간단히 만든 String입니다. 대강 20자 쯤 해요."
DEFAULT_CONVERTED_FILE_NAME = "converted.txt"
DEFAULT_INPUT_TEXT_FILE_NAME = "input.txt"



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Font shuffler & text converter using custom font dictionary.")
    subparsers = parser.add_subparsers(dest = "command", required=True)

    parser_convert = subparsers.add_parser('shuffle', help="Font Shuffler")
    parser_convert.add_argument("-t","--target-font",action="store",required=True,type=str,help="Path to the original font (e.g., TTF/OTF) to be shuffled.")
    parser_convert.add_argument("-n","--font-name",action="store",default=DEFAULT_TARGET_FONT_NAME,type=str,help="Name of the output font after shuffling (default: ShuffledFont).")
    parser_convert.add_argument("-e","--ext-font-file",action="store",default=DEFAULT_TARGET_EXT_FONT,type=str,help="Filename to export the shuffled font file (default: output.ttf).")
    parser_convert.add_argument("-d","--ext-dict-name",action="store",default=DEFAULT_CONVERT_DICT_NAME,type=str,help="Filename to export the font character mapping dictionary (default: convert_map.json).")

    parser_transform = subparsers.add_parser("convert", help="Convert text using a pre-generated font dictionary.")
    parser_transform.add_argument("-n","--inline",action="store_true",default=False,help="Interpret -i/--input as direct text instead of a file path.") 
    parser_transform.add_argument("-i","--input",action="store",required=True,type=str,help="Input text (if --inline) or path to input file (if not).")
    parser_transform.add_argument("-o","--output",action="store",default=DEFAULT_CONVERTED_FILE_NAME, type=str,help="Path to save the converted result (default: converted.txt). Only used if not --inline.")
    parser_transform.add_argument("-d","--target-dict",action="store",default=DEFAULT_CONVERT_DICT_NAME,type=str,help="Path to the character mapping dictionary file (default: convert_map.json).")
    parser_transform.add_argument("-c","--compare",action="store_true",default = False,help="Include original text alongside converted result for comparison.")
    

    args = parser.parse_args()

    if args.command == "shuffle":
        fc = FontConverter(args.target_font)
        fc.shuffle()
        fc.export(args.font_name, args.ext_font_file, args.ext_dict_name)
        print("FONT SHUFFLED")
    elif args.command == "convert":
        fca = TextConverter(args.target_dict)
        if(args.inline):
            if not args.compare:
                print(fca.convert_text(args.input))
            else:
                print(f"original: {args.input}\nconverted: {fca.convert_text(args.input)}")
        else:
            target_text=""
            with open(args.input,"r",encoding='utf-8') as f:
                target_texts = f.readlines()
                for _ in target_texts:
                    target_text+=_
            converted_text = fca.convert_text(target_text)
            outtext = None
            if(args.compare):
                outtext = f"original: {target_text}\n\nconverted: {converted_text}"
            else:
                outtext = converted_text

            with open(args.output, "w",encoding='utf-8') as f:
                f.write(outtext)
            print("File Written")
