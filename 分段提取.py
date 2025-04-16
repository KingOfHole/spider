import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import os
import re


def sanitize_filename(filename):
    """清理文件名中的非法字符"""
    if not filename:  # 如果标题为空
        filename = "无标题"
    # 替换Windows文件名中不允许的字符
    filename = re.sub(r'[\\/*?:"<>|]', "_", filename)
    # 去除首尾空格
    filename = filename.strip()
    # 限制文件名长度（Windows最大255字符）
    return filename[:200]


def get_news_links(list_url):
    """获取新闻链接列表"""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    try:
        response = requests.get(list_url, headers=headers, timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')
        links = []

        for a_tag in soup.select('.fixList a[href]'):
            href = a_tag.get('href')
            if href:  # 确保href存在
                if not href.startswith('http'):
                    href = urljoin(list_url, href)
                links.append(href)
        return links
    except Exception as e:
        print(f"获取链接失败: {e}")
        return []


def parse_news_page(news_url):
    """解析单篇新闻内容"""
    try:
        response = requests.get(news_url, timeout=10)
        response.raise_for_status()
        response.encoding = 'utf-8'
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.find('h1').get_text(strip=True) if soup.find('h1') else "无标题"
        title = sanitize_filename(title)

        content_div = soup.find('div', class_='article') or soup.find('div', id='artibody')
        if content_div:
            content = '\n'.join([p.get_text(strip=True) for p in content_div.find_all('p') if p.get_text(strip=True)])
        else:
            content = "无正文"

        return {
            'title': title,
            'content': content,
            'url': news_url
        }
    except Exception as e:
        print(f"解析页面失败: {news_url}, 错误: {e}")
        return None


def save_individual_txt(article, output_dir="news_articles"):
    """将单篇新闻保存为单独的TXT文件"""
    if not article:  # 检查article是否为None
        print("错误: 传入的article为None")
        return None

    try:
        os.makedirs(output_dir, exist_ok=True)

        # 确保title存在且不为空
        title = article.get('title', '无标题')
        filename = f"{title}.txt"
        filepath = os.path.join(output_dir, filename)

        counter = 1
        while os.path.exists(filepath):
            filename = f"{title}_{counter}.txt"
            filepath = os.path.join(output_dir, filename)
            counter += 1

        with open(filepath, 'w', encoding='utf-8-sig') as f:
            f.write(f"标题: {title}\n")
            f.write(f"URL: {article.get('url', '无URL')}\n")
            f.write("正文:\n")
            f.write(article.get('content', '无正文'))

        print(f"已保存: {filename}")
        return filepath
    except Exception as e:
        print(f"保存文件失败: {e}")
        return None


def crawl_sina_military():
    list_url = 'https://mil.news.sina.com.cn/roll/index.d.html?cid=57919'
    news_links = get_news_links(list_url)

    for link in news_links[:3]:  # 测试3篇
        print(f"正在抓取: {link}")
        article = parse_news_page(link)
        if article:  # 只有当article不是None时才保存
            save_individual_txt(article)
        else:
            print(f"跳过无效文章: {link}")
        time.sleep(2)


if __name__ == '__main__':
    crawl_sina_military()