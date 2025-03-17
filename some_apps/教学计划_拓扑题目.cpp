#include<iostream>
#include<vector>
#include<string>
#include<queue>
#include<unordered_map>
using namespace std;

//代码注释集中在在函数实现部分
struct course {
	string name;
	int credit;
	vector<course> previous;
	int aftercredit=credit;//将后续有关课程学分汇总，较高者假设为基础课程，较低者假设为专业课程；
	double index1,index2;//分别对应两个策略拓扑排序时的依据指数；
	course(string a="\0", int b=0, vector<course> c={})
		:name(a), credit(b), previous(c) {}
	void indexcount();
};
bool operator ==(course a, course b);
struct graph {
	int vexsnum, allcredit = 0, returnjudge = 0;
	vector<course> vexs,toplist;
	vector<vector<int>> arcs;
	void AfterCredit(int last);
	graph(int a, vector<course> c);
	void topqueue(int choice);
};
bool check(vector<course> list);
bool errorcheck(vector<course> list, int average, int& error, int creditlimit);
int creditsum(vector<course> list);
vector< vector<course>> design(graph &g, int termlimit, int creditlimit, int choice, vector<int>& error);
int prefind(vector<course> vexs, string prename);
int main()
{
	int termlimit, creditlimit,coursenum,choice;
	vector< vector<course>> schedule;
	cout << "请设置学期总数："<<endl;
	cin >> termlimit;
	cout << "请设置学分上限："<<endl;
	cin >> creditlimit;
	cout << "请输入专业课程总数："<<endl;
	cin >> coursenum;
	cout << "请依次输入课程信息："<<endl;
	vector<course> vexs;
	vector< vector<string>> prename;
	for (int i = 0; i < coursenum; i++)
	{
		string name;
		int prenum;
		vector<string> singleprename;
		int credit;
		vector<course> previous;
		cout << "请输入课程名称：";
		cin >> name;
		cout << "请输入课程学分：";
		cin >> credit;
		cout << "请输入本课程先修课程数目：";
		cin >> prenum;
		if (prenum != 0)
		{
			cout << "请输入先修课程名称（备注:以空格分隔）：";
			for (int i = 0; i < prenum; i++)
			{
				string n;
				cin >> n;
				singleprename.push_back(n);
			}
		}
		prename.push_back(singleprename);
		course cou(name, credit, previous);
		vexs.push_back(cou);
	}
	//将先修课程根据名称加入pre数组中，同时判断有无先修课程非此专业必修课程情况；
	for (int i = 0; i < coursenum; i++)
	{
		for(int j=0;j<prename[i].size(); j++)
		{
			if (!vexs.empty())
			{
				if (prefind(vexs, prename[i][j]) == -1)
				{
					cout << "错误加入了非本专业学习的先修课程！";
					return 0;
				}
				vexs[i].previous.push_back(vexs[prefind(vexs,prename[i][j])]);
			}
		}
	}
	cout << "请输入选择的策略：";
	cout << "1--学业负担均匀分布：";
	cout << "2--课程集中于前期：";
	cin >> choice;
	vector<int> error;
	graph DAG(coursenum, vexs);
	if (DAG.returnjudge)return 0;
	DAG.topqueue(choice);
	if (DAG.returnjudge)return 0;
	schedule = design(DAG, termlimit, creditlimit, choice,error);
	if (DAG.returnjudge)return 0;
	double variance=0;
	for (int i = 0; i < schedule.size(); i++)
	{
		cout << "学期"<<i+1<<" ：";
		for(int j=0;j<schedule[i].size();j++)
		{
			cout << schedule[i][j].name<<" ";
		}
		if(choice==2)cout << endl;
		if (choice == 1)
		{
			cout << "离学分均值距离：" << error[i];
			cout << endl;
			variance += error[i] * error[i];
		}
	}
	if(choice==1)
	cout << "方差为：" << variance / schedule.size();
	return 0;
}
//函数实现：
auto cmp1 = [](course a, course b) {return a.index1 < b.index1; };
auto cmp2 = [](course a, course b) {return a.index2 < b.index2; };
void course::indexcount() {
	if (previous.size() == 0)
	{
		index2 = INT_MAX;
		index1 = INT_MAX;
		return;
	}
	index2 = 1 / (previous.size() * credit);//选择策略2尽量集中时只需要确保先修课程尽量少，同时自己的学分也尽量小一点，满足在学分上限内；
	index1 = aftercredit / (previous.size() * previous.size());//选择策略1需要计算出重要度，根据自己的后续课程的学分和前置课程数目来确定；
}
bool operator ==(course a, course b)
{
	return a.name == b.name;
}
void graph::AfterCredit(int last)//递归计算后续课程总分来计算重要性；
{
	for (int i = 0; i < vexsnum; i++)
	{
		if (arcs[i][last] != 0)
		{
			vexs[i].aftercredit += vexs[last].aftercredit;
			AfterCredit(i);
		}
	}
	return;
}
graph::graph(int a, vector<course> c) :vexsnum(a), vexs(c) {
	for (int i = 0; i < vexsnum; i++)
	{
		vector<int> v(vexsnum);
		arcs.push_back(v);
	}
	//扩容，避免溢出
	//初始化邻接矩阵；
	for (int i = 0; i < vexsnum; i++)
	{
		allcredit += vexs[i].credit;//统计总学分，为平均计算准备；
		for (int j = 0; j < vexsnum; j++)
		{
			if (!vexs[i].previous.empty())
			{
				if (find(vexs[i].previous.begin(), vexs[i].previous.end(), vexs[j]) != vexs[i].previous.end())
				{
					arcs[j][i] = 1;
				}
			}
		}
	}
	//计算 aftercredit(由后至前计算)同时判断有无大环；
	vector<int> last;
	for (int i = 0; i < vexsnum; i++)
	{
		int count = 0;
		for (int j = 0; j < vexsnum; j++)
		{
			if (arcs[i][j] != 0)
			{
				count++;
				break;
			}
		}
		if (count == 0)
			last.push_back(i);
	}
	if (last.size() == 0)
	{
		cout << "该先修课程关系图存在大环，故无解";
		returnjudge = 1;
	}
	for (int i = 0; i < last.size(); i++)
	{
		AfterCredit(last[i]);
	}
}
void graph::topqueue(int choice)//based on index ,用优先队列根据两个指数将拓扑排序时置于前面；
{
	for (int i = 0; i < vexsnum; i++)
		vexs[i].indexcount();
	priority_queue<course, vector<course>, decltype(cmp1)> que1(cmp1);
	priority_queue<course, vector<course>, decltype(cmp2)> que2(cmp2);
	vector<int> indegree;
	unordered_map<string, int> visited;
	for (int i = 0; i < vexsnum; i++)
	{
		visited[vexs[i].name] = 1;
		int count = 0;
		for (int j = 0; j < vexsnum; j++)
		{
			count += arcs[j][i];
		}
		indegree.push_back(count);
		if (count == 0)
		{
			que1.push(vexs[i]);
			que2.push(vexs[i]);
			visited[vexs[i].name] = 0;
		}
	}
	while (!que1.empty() && !que2.empty())
	{
		course current = choice == 1 ? que1.top() : que2.top();
		toplist.push_back(current);
		if (choice == 1)que1.pop();
		else  que2.pop();
		for (int i = 0; i < vexsnum; i++)
		{
			if (!vexs[i].previous.empty())
			{
				if (find(vexs[i].previous.begin(), vexs[i].previous.end(), current) != vexs[i].previous.end())
					indegree[i]--;
			}
			if (indegree[i] == 0 && visited[vexs[i].name])
			{
				if (choice == 1)que1.push(vexs[i]);
				else que2.push(vexs[i]);
				visited[vexs[i].name] = 0;
			}
		}
	}
	for (int i = 0; i < vexsnum; i++)
	{
		if (visited[vexs[i].name])
		{
			cout << "存在回路";
			returnjudge = 1;
			break;
		}
	}
}
bool check(vector<course> list)//检查一学期内是否有互为先修课程情况
{
	unordered_map<string, int> exitence;
	for (int i = 0; i < list.size(); i++)
	{
		exitence[list[i].name] = 1;
		for (course j : list[i].previous)
		{
			if (exitence[j.name] == 1)
				return false;
		}
	}
	return true;
}
bool errorcheck(vector<course> list, int average, int& error, int creditlimit)//辅助计算最小距离并时刻更新最小距离
{

	int count = 0;
	for (course i : list)
		count += i.credit;
	if ((abs(count - average) < error) && check(list) && count <= creditlimit)
	{
		error = abs(count - average);
		return true;
	}
	else
		return false;
}
int creditsum(vector<course> list)//求和计算学分
{
	int sum = 0;
	for (course i : list)
		sum += i.credit;
	return sum;
}
vector< vector<course>> design(graph &g, int termlimit, int creditlimit, int choice, vector<int>& error)//将拓扑排序根据要求划分为以学期为单位的课程数组
{
	vector< vector<course>> table;
	if (choice == 1)
	{
		int average = g.allcredit / termlimit;//下取整误差存在；
		int term = 0;
		int flag = 0;
		while (term <= termlimit && flag < g.toplist.size())
		{
			int err = INT_MAX;
			int i = 0;
			vector<course> current(g.toplist.begin() + flag, g.toplist.begin() + flag + i + 1);
			while (errorcheck(current, average, err, creditlimit))//找出满足误差条件，满足先修条件，满足学分限制条件i；
			{
				if (flag + i + 1 < g.toplist.size())
				{
					i++;
					current.assign(g.toplist.begin() + flag, g.toplist.begin() + flag + i + 1);
				}
				else
				{
					i++;//为了和check错误的i均为正解i+1; 
					break;
				}
			}
			i--;
			if (i != -1)
			{
				vector<course> current(g.toplist.begin() + flag, g.toplist.begin() + flag + i + 1);
				table.push_back(current);
				flag += i + 1;
			}
			term++;
			error.push_back(err);
		}
		if (term <= termlimit)
			return table;
		//走到这里说明100误差失败，则可认定无法在规定trem内完成学业分配；
		g.returnjudge = 1;
		cout << "无法在规定trem内完成学业分配";
		return table;
	}
	if (choice == 2)
	{
		int term = 0;
		int flag = 0;
		while (term < termlimit && flag < g.toplist.size())
		{
			int i = 0;
			vector<course> current(g.toplist.begin() + flag, g.toplist.begin() + flag + i + 1);
			while (check(current) && creditsum(current) <= creditlimit)//找出满足先修条件和学分限制条件i；
			{
				if (flag + i + 1 < g.toplist.size())
				{
					i++;
					current.assign(g.toplist.begin() + flag, g.toplist.begin() + flag + i + 1);
				}
				else
				{
					i++;//为了和check错误的i均为正解i+1; 
					break;
				}
			}
			i--;
			if (i != -1)
			{
				vector<course> current(g.toplist.begin() + flag, g.toplist.begin() + flag + i + 1);
				table.push_back(current);
				flag += i + 1;
			}
			term++;
		}
		if (flag < g.toplist.size())//说明是term超过了；
		{
			g.returnjudge = 1;
			cout << "无法在规定trem内完成学业分配";
			return table;
		}
		else	//说明分配成功；
			return table;
	}
}
int prefind(vector<course> vexs, string prename)
{
	for (int i = 0; i < vexs.size(); i++)
	{
		if (vexs[i].name == prename)
			return i;
	}
	return -1;
}
