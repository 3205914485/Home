#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<queue>
#include<unordered_map>
#include <bitset>
 /*
 蛁嚙踝蕭1嚙踝蕭嚙踝蕭嚙豌�嚙緣xt嚙衝潘蕭揤嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭荎嚙衝橘蕭盓嚙誰�嚙踝蕭嚙踝蕭嚙諂槽對蕭嚙踝蕭嚙�60-70%嚙踝蕭
	 2嚙踝蕭嚙踝蕭嚙豌梧蕭蛁:嚙踝蕭嚙賢髡嚙豌喉蕭嚙踝蕭嚙諸橘蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙誼湛蕭嚙踝蕭嚙踝蕭蛁
		1嚙踝蕭嚙踝蕭嚙踝蕭盓嚙踝蕭嚙諂踝蕭嚙衝潘蕭嚙踝蕭蘇嚙踝蕭繚嚙踝蕭嚙諸賂蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙誹喉蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙豬蜊�嚙踝蕭嚙�
		2嚙踝蕭揤嚙踝蕭奀嚙踝蕭堈苤嚙誹踝蕭揤奀嚙賭ㄛ嚙踝蕭嚙踝蕭峈嚙踝蕭揤奀嚙賣摯n^2嚙踝蕭嚙誰蛛蕭嚙踝蕭猀嚙�
		3嚙踝蕭嚙誼喉蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭痗嚙踝蕭嚙踝蕭嚙踝蕭躁嚙踝蕭觸嚙踝蕭嚙踝蕭嚙�
		4嚙踝蕭嚙誼唳掛嚙踝蕭嚙踝蕭2.0嚙踝蕭嚙踝蕭嚙踝蕭揤嚙踝蕭嚙褓�嚙踝蕭嚙踝蕭驍嚙�01嚙賣揣嚙踝蕭嚙諛蛛蕭嚙諍賂蕭0嚙踝蕭1嚙褓脹�嚙踝蕭嚙踝蕭嚙踝蕭嚙誰湛蕭郱嚙褊踝蕭嚙踝蕭嚙踝蕭嚙練ain嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭妗嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭硒嚙請�嚙踝蕭嚙�
		嚙誕莎蕭嚙誰蛛蕭嚙衝潘蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙衝潘蕭嚙踝蕭嚙踝蕭嚙衝潘蕭芛嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭奀嚙踝蕭嚙誕蛛蕭嚙踝蕭侅嚙誰鳴蕭亶嚙踝蕭硒佼嚙踝蕭嚙踝蕭瘙橘蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙課閡鳴蕭驍嚙踝蕭嚙�
	 3嚙踝蕭Hffman-Compressor 2.0.exe嚙踝蕭嚙踝蕭奀嚙賭ㄩ2023.6.22;
*/
using namespace std;
typedef int Status;
#define OK 1
#define ERROR 0
struct treenode {
	char element;
	int weight;
	treenode* left;
	treenode* right;
	treenode(int a,char ele=0,treenode* l = NULL, treenode* r = NULL) :weight(a),element(ele), left(l), right(r) {}
};
auto compTreenode = [](treenode* a, treenode* b) {return a->weight > b->weight; };
treenode* Haffman(priority_queue<treenode*, vector<treenode*>, decltype(compTreenode)>& forest)//嚙踝蕭嚙踝蕭嚙踝蕭苤嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭
{
	while (forest.size() >= 2)
	{
		treenode* l = forest.top();
		forest.pop();
		treenode* r = forest.top();
		forest.pop();
		treenode* newtree = new treenode(l->weight + r->weight,0, l, r);
		forest.push(newtree);
	}
	return forest.top();
}
Status DFSCoding(treenode* root, unordered_map<char, string>&codes,string &s)//01嚙踝蕭嚙踝蕭硉
{
	if (root->left)
	{
		s += "0";
		DFSCoding(root->left, codes, s);
	}
	if (root->right)
	{
		s += "1";
		DFSCoding(root->right, codes, s);
	}
	if (!root->right && !root->left)codes[root->element] = s;
	s = s.substr(0, s.size() - 1);
	return OK;
}
Status Write_Code(int &stop, unordered_map<char, string>codes, string bit_str,string name="D:/data.bin", string dicname="D:/dics.txt")//嚙衝鳴蕭迡嚙踝蕭
{ 
	//嚙踝蕭珨嚙踝蕭嚙誰�嚙踝蕭嚙踝蕭硉嚙踝蕭嚙趟ㄩ
	ofstream dicfile(dicname,ios::app);//蘇嚙誕湛蕭嚙踝蕭嚙踝蕭
	if (!dicfile.is_open())
	{
		std::cerr << "Fail to open file!" << std::endl;
		stop = 1;return -1;	
	}
	dicfile << ' ';
	for (auto i : codes)
	{
		dicfile << i.first<< i.second<<' ';
	}
	dicfile.close();
	//嚙誹塚蕭嚙踝蕭嚙誰�嚙踝蕭嚙諂湛蕭躁嚙�01嚙踝蕭嚙踝蕭嚙複湛蕭嚙踝蕭

	// 嚙踝蕭 01 嚙踝蕭藩 8 嚙踝蕭嚙誰瘀蕭蛌峈珨嚙踝蕭嚙誰誹�嚙踝蕭嚙諄迎蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙�
	vector<u_int8_t> data;
	for (size_t i = 0; i < bit_str.size(); i += 8) {
		if (i == bit_str.size() - 13)
			;
		std::string sub_str = bit_str.substr(i, 8);
		u_int8_t byte_data = std::bitset<8>(sub_str).to_ulong();
		data.push_back(byte_data);
	}
	int n = 8-bit_str.size() % 8;  // 嚙踝蕭嚙踝蕭弇嚙衝賂蕭嚙踝蕭
	data[data.size() - 1] <<=n;//籵嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭8弇奀嚙衝喉蕭0嚙踝蕭嚙賤ㄛ嚙諉塚蕭嚙誹踝蕭揤奀嚙踝蕭嚙踝蕭嚙踝蕭嚙�0弇嚙踝蕭�伐蕭嚙�
	char byte = n ;  // 嚙踝蕭嚙踝蕭嚙諄鳴蕭嚙諄� 1嚙踝蕭嚙踝蕭嚙踝蕭弇峈 0
	// 嚙踝蕭醴嚙踝蕭嚙衝潘蕭
	std::ofstream outfile(name, std::ios::binary);
	if (!outfile.is_open()) {
		std::cerr << "Fail to open file!" << std::endl;
		stop = 1; return -1;
	}
	outfile.write(&byte, 1);  // 嚙誹塚蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭迡嚙踝蕭嚙踝蕭祧嚙踝蕭嚙�
	// 嚙踝蕭嚙踝蕭嚙踝蕭嚙諂塚蕭嚙踝蕭嚙複賂蕭宒迡嚙踝蕭嚙衝潘蕭
	outfile.write(reinterpret_cast<const char*>(data.data()), data.size());

	outfile.close();
	return OK;
}
Status Decode(string dename,string binname="D:/data.bin", string dicsname = "D:/dics.txt")//嚙衝潘蕭嚙踝蕭嚙踝蕭
{
	//嚙踝蕭龰嚙誰萎ㄩ
	ifstream dicfile(dicsname);
	if (!dicfile.is_open()) {
		std::cerr << "Fail to open file!" << std::endl;
		return -1;
	}
	unordered_map<char, string> codes;
	char w,ns='t';
	string s;
	while (dicfile.get(w))
	{
		if (w == ' ')
		{
			codes[ns] = s;
			ns=dicfile.get();
			s = "";
			continue;
		}
		s += w;
	}
	dicfile.close();
	//嚙踝蕭洷嚙踝蕭嚙踝蕭蛌
	unordered_map<string, char>dics;
	for (auto n : codes)
	{
		dics[n.second] = n.first;
	}
	ifstream infile(binname, std::ios::binary);
	if (!infile.is_open()) {
		std::cerr << "Fail to open file!" << std::endl;
		return -1;
	}

	// 嚙踝蕭龰嚙衝潘蕭嚙衝湛蕭苤
	infile.seekg(0, std::ios::end);
	size_t file_size = infile.tellg();
	infile.seekg(0, std::ios::beg);

	// 嚙踝蕭嚙踝蕭嚙褓∴蕭祧琭嚙踝蕭嚙衛選蕭嚙踝蕭祧嚙論迎蕭嚙諄� 8 弇嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭
	vector<u_int8_t> data(file_size);
	infile.read(reinterpret_cast<char*>(data.data()), file_size);
	// 嚙踝蕭龰嚙踝蕭嚙踝蕭弇嚙衝賂蕭嚙踝蕭
	int n = (data[0]);
	string bit_str;
	for (size_t i = 1; i < data.size(); i++) {
		bit_str += std::bitset<8>(data[i]).to_string();
	}	
	bit_str = bit_str.substr(0, bit_str.size() - n);
	// 嚙諍梧蕭嚙踝蕭嚙踝蕭嚙踝蕭
	infile.close();	

	cout << "Loading......" <<endl;
	ofstream outfile(dename, ios::app);
	string popo;
	for (auto s : bit_str)
	{
		popo += s;
		if (dics.find(popo)!=dics.end())
		{
			outfile << dics.find(popo)->second;
			popo.clear();
		}
	}
	outfile.close();
	return OK;
}
Status Read(int &stop,string name, unordered_map<char,int> &dic, priority_queue<treenode*, vector<treenode*>, decltype(compTreenode)> &forest,vector<char>&text)
{
	ifstream fin(name, ios::in);
	if (!fin)
	{
		stop = 1;
		cout << "嚙衝潘蕭嚙踝蕭龰囮嚙豌�嚙�";
		return ERROR;
	}
	char c; int u = 0;
	while ((c = fin.get()) != EOF)
	{
		dic[c]++;
		text.push_back(c);
	}
	for (auto p:dic)
	{
		char element = p.first; int weight = p.second;
		treenode* place = new treenode(weight,element);
		forest.push(place);
	}
	return OK;
}
int main()
{
	while(1)
	{

		cout << "嚙踝蕭嚙踝蕭嚙踝蕭揤嚙踝蕭嚙踝蕭2.0" << endl;
		cout << "嚙踝蕭嚙踝蕭1嚙踝蕭揤嚙踝蕭醴嚙踝蕭嚙衝潘蕭嚙踝蕭嚙踝蕭嚙踝蕭1嚙踝蕭" << endl;
		cout << "嚙踝蕭嚙踝蕭2嚙踝蕭嚙踝蕭揤醴嚙踝蕭嚙衝潘蕭嚙踝蕭嚙踝蕭嚙踝蕭2嚙踝蕭" << endl;
		cout << "嚙誼喉蕭嚙踝蕭嚙踝蕭嚙踝蕭3 " << endl;
		int choice = 1;
		cin >> choice;
		string dicname, binname;
		if (choice == 1)
		{
			cout << "嚙踝蕭嚙踝蕭醴嚙踝蕭嚙衝潘蕭嚙踝蕭硊嚙踝蕭嚙踝蕭嚙踝蕭繚嚙踝蕭嚙踝蕭:";
			unordered_map<char, int> dic;
			vector<char> text;
			priority_queue<treenode*, vector<treenode*>, decltype(compTreenode)> forest(compTreenode);
			int stop = 0;
			string txtname;
			cin >> txtname;
			Read(stop, txtname, dic, forest, text);//嚙踝蕭龰ⅰ嚙褓�嚙�
			if (stop)return 0;
			treenode* root = Haffman(forest);
			unordered_map<char, string>codes;//嚙踝蕭嚙趟掛嚙踝蕭
			string temp;
			DFSCoding(root, codes, temp);
			string translated;
			for (auto c : text)
				translated += codes[c];
			cout << "蘇嚙踝蕭醴嚙踝蕭揤嚙踝蕭嚙踝蕭硊嚙踝蕭D:/data.bin & D:/dics.txt(嚙誰梧蕭瘣Ｆ湛蕭躁嚙踝蕭芵嚙踝蕭嚙諸對蕭嚙踝蕭躁嚙�); 嚙踝蕭嚙踝蕭嚙衝選蕭嚙諸對蕭嚙踝蕭嚙誰瘀蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙�:1 嚙踝蕭嚙踝蕭嚙踝蕭嚙趟ㄩ0 嚙踝蕭" << endl;
			int choice2;
			cin >> choice2;
			if (choice2 == 1)	
			{
				cout << "嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭嚙踝蕭硊嚙踝蕭";
				cin >> binname >> dicname;
				Write_Code(stop, codes, translated, binname, dicname);
			}
			else Write_Code(stop, codes, translated);
			if (stop)return 0;
		}
		if (choice == 2)
		{
			cout << "嚙踝蕭嚙踝蕭嚙踝蕭嚙諸對蕭躁嚙踝蕭嚙誰�(埭嚙衝潘蕭嚙踝蕭硊嚙踝蕭嚙踝蕭嚙踝蕭嚙衝潘蕭嚙踝蕭硊嚙踝蕭): ";
			cin >> binname >> dicname;
			cout << "嚙踝蕭嚙踝蕭嚙諸對蕭嚙踝蕭嚙誰瘀蕭嚙�";
			string name2;
			cin >> name2;
			Decode(name2, binname, dicname);
		}
		if (choice == 3)return 0;
	}
	return 0;
}		
