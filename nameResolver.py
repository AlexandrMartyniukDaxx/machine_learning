
titles = ['Mr.', 'Mrs.', 'Lady.', 'Miss.', 'Mlle.', 'Col.', 'Capt.', 'Dr.', 'Master.', 'Rev.']


def get_name(input_string):
    title = None
    index = -1

    for _title in titles:
        index = input_string.find(_title)
        if index != -1:
            title = _title
            break

    if index == -1:
        second_part = input_string.split(', ', 2)[1]
        if second_part:
            if '(' in second_part:
                return second_part.split('(', 2)[1].split(' ', 2)[0]
            else:
                return second_part.split(' ')[0]
        else:
            return input_string

    if title is not None:
        if title in ['Mrs.', 'Lady.']:
            brace_ind = input_string.find('(', index)
            if brace_ind == -1:
                return input_string[index:].split(' ', 2)[1]
            else:
                return input_string[brace_ind + 1:].split(' ', 1)[0].strip('.")')
        else:
            return input_string[index:].split(' ', 2)[1]

    other = input_string[index:].split(' ', 2)[1]
    if other:
        return other.strip('().')
    else:
        return input_string
