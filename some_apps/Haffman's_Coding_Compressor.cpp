#include<iostream>
#include<fstream>
#include<vector>
#include<string>
#include<queue>
#include<unordered_map>
#include <bitset>
 /*
注：  	 1、介绍：txt文件压缩器；（中英文均支持），大约压缩比60-70%；是我在数据结构课后自己设计的小程序。
	 	2、功能备注:具体功能程序已经给出，此处仅备注
			1、代码支持自建文件，默认路径已给出（可在程序内自行修改）；
			2、压缩时间远小于解压时间，是因为解压时涉及n^2的字典查找；
			3、此程序核心在于二进制文件的操作；
			4、此版本仅是2.0，还有压缩比（如优化01存储，折叠重复0、1等等）、程序执行方式（设置main函数参数，实现命令行执行）、
			合并字典文件与数据文件（加文件头描述）。但由于时间关系，此处只能呈现此初级版本。后续会再一一优化。
	 	3、Hffman-Compressor 2.0.exe设计时间：2023.6.22;
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
treenode* Haffman(priority_queue<treenode*, vector<treenode*>, decltype(compTreenode)>& forest)//构建最小生成树；
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
Status DFSCoding(treenode* root, unordered_map<char, string>&codes,string &s)//01串赋值
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
Status Write_Code(int &stop, unordered_map<char, string>codes, string bit_str,string name="D:/data.bin", string dicname="D:/dics.txt")//文化写入
{ 
	//第一部分：将字典存入：
	ofstream dicfile(dicname,ios::app);//默认创立；
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
	//第二部分：将源文件01二进制存入

	// 将 01 串每 8 个字符转为一个字节，存为二进制数
	vector<u_int8_t> data;
	for (size_t i = 0; i < bit_str.size(); i += 8) {
		if (i == bit_str.size() - 13)
			;
		std::string sub_str = bit_str.substr(i, 8);
		u_int8_t byte_data = std::bitset<8>(sub_str).to_ulong();
		data.push_back(byte_data);
	}
	int n = 8-bit_str.size() % 8;  // 补余位的个数
	data[data.size() - 1] <<=n;//通过左移运算符解决不足8位时的充0问题，从而在解压时将不足的0位舍去；
	char byte = n ;  // 将最高位设为 1，其他位为 0
	// 打开目标文件
	std::ofstream outfile(name, std::ios::binary);
	if (!outfile.is_open()) {
		std::cerr << "Fail to open file!" << std::endl;
		stop = 1; return -1;
	}
	outfile.write(&byte, 1);  // 第二个参数是写入的字节数
	// 将数据以二进制格式写入文件
	outfile.write(reinterpret_cast<const char*>(data.data()), data.size());

	outfile.close();
	return OK;
}
Status Decode(string dename,string binname="D:/data.bin", string dicsname = "D:/dics.txt")//文件解码
{
	//获取字典：
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
	//哈希表反转
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

	// 获取文件的大小
	infile.seekg(0, std::ios::end);
	size_t file_size = infile.tellg();
	infile.seekg(0, std::ios::beg);

	// 逐个读取字节，将每个字节转换为 8 位二进制数
	vector<u_int8_t> data(file_size);
	infile.read(reinterpret_cast<char*>(data.data()), file_size);
	// 获取补余位的个数
	int n = (data[0]);
	string bit_str;
	for (size_t i = 1; i < data.size(); i++) {
		bit_str += std::bitset<8>(data[i]).to_string();
	}	
	bit_str = bit_str.substr(0, bit_str.size() - n);
	// 关闭输入流
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
		cout << "文件读取失败！";
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

		cout << "哈夫曼压缩器2.0" << endl;
		cout << "功能1：压缩目标文件，输入1：" << endl;
		cout << "功能2：解压目标文件，输入2：" << endl;
		cout << "退出：输入3 " << endl;
		int choice = 1;
		cin >> choice;
		string dicname, binname;
		if (choice == 1)
		{
			cout << "输入目标文件地址（绝对路径）:";
			unordered_map<char, int> dic;
			vector<char> text;
			priority_queue<treenode*, vector<treenode*>, decltype(compTreenode)> forest(compTreenode);
			int stop = 0;
			string txtname;
			cin >> txtname;
			Read(stop, txtname, dic, forest, text);//读取频度；
			if (stop)return 0;
			treenode* root = Haffman(forest);
			unordered_map<char, string>codes;//密码本；
			string temp;
			DFSCoding(root, codes, temp);
			string translated;
			for (auto c : text)
				translated += codes[c];
			cout << "默认目标压缩地址：D:/data.bin & D:/dics.txt(分别存储源文件和辅助压缩文件); 如果有目标压缩地址，请输入:1 否则输入：0 ：" << endl;
			int choice2;
			cin >> choice2;
			if (choice2 == 1)	
			{
				cout << "请输入两个地址：";
				cin >> binname >> dicname;
				Write_Code(stop, codes, translated, binname, dicname);
			}
			else Write_Code(stop, codes, translated);
			if (stop)return 0;
		}
		if (choice == 2)
		{
			cout << "输入需解压文件地址(源文件地址，辅助文件地址，): ";
			cin >> binname >> dicname;
			cout << "输入解压至地址：";
			string name2;
			cin >> name2;
			Decode(name2, binname, dicname);
		}
		if (choice == 3)return 0;
	}
	return 0;
}								
