-- 将 markdown 标题层级上移一级：##(2)->\section(1)，###(3)->\subsection(2)
-- 报告正文无 "#"（一级标题已在预处理中移除），故 level>=2 整体 -1 安全。
function Header(el)
  if el.level >= 2 then
    el.level = el.level - 1
  end
  return el
end
